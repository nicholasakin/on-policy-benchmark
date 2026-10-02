#!/usr/bin/env python3
"""Schedule reproducible Simple Spread experiments, one process group per GPU."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent


def stamp():
    return datetime.now(timezone.utc).isoformat()


def plan(args, batch):
    workers = {'reference': [128], 'workers': [4, 16, 32], 'smoke': [4]}[args.suite]
    steps = args.steps or (200 if args.suite == 'smoke' else 20_000_000)
    jobs = []
    for seed in args.seeds:
        for count in workers:
            for algo in ['ippo', 'mappo']:
                name = f'{batch}_{algo}_w{count}_seed{seed}'
                interval = max(1, 80_000 // (25 * count)) if args.suite != 'smoke' else 1
                command = [sys.executable, '-u', str(ROOT / 'onpolicy/scripts/train/train_mpe.py'),
                    '--env_name', 'MPE', '--algorithm_name', 'rmappo',
                    '--experiment_name', name, '--scenario_name', 'simple_spread',
                    '--num_agents', '3', '--num_landmarks', '3', '--seed', str(seed),
                    '--n_training_threads', '1', '--n_rollout_threads', str(count),
                    '--episode_length', '25', '--num_env_steps', str(steps),
                    '--ppo_epoch', '10', '--num_mini_batch', '1', '--use_ReLU',
                    '--gain', '0.01', '--lr', '7e-4', '--critic_lr', '7e-4',
                    '--use_eval', '--n_eval_rollout_threads', '2' if args.suite == 'smoke' else '20',
                    '--eval_interval', str(interval), '--use_wandb', '--user_name', 'reproduction']
                if algo == 'ippo':
                    command.append('--use_centralized_V')
                if args.cpu:
                    command.append('--cuda')
                if steps < 25 * count:
                    raise ValueError(f'{steps} steps cannot fill one rollout with {count} workers')
                jobs.append(dict(name=name, algorithm=algo, seed=seed, workers=count,
                    requested_steps=steps, actual_steps=steps // (25 * count) * (25 * count),
                    command=command, status='pending',
                    result_dir=str(ROOT / 'onpolicy/scripts/results/MPE/simple_spread/rmappo' / name)))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['reference', 'workers', 'smoke'], default='reference')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--gpus', nargs='+', default=['0', '1'], help='CUDA device identifiers, one job each')
    parser.add_argument('--steps', type=int, help='Override per-run environment-step budget')
    parser.add_argument('--cpu', action='store_true', help='Use one CPU job slot (useful for smoke tests)')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without creating files or training')
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.gpus)) != len(args.gpus):
        parser.error('Seeds and GPU identifiers must be unique')
    if any(seed < 0 for seed in args.seeds) or (args.steps is not None and args.steps <= 0):
        parser.error('Seeds must be nonnegative and steps must be positive')
    batch = f'{args.suite}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:6]}'
    try:
        jobs = plan(args, batch)
    except ValueError as exc:
        parser.error(str(exc))
    slots = ['cpu'] if args.cpu else args.gpus
    print(f'{args.suite}: {len(jobs)} runs; {len(slots)} concurrent jobs; '
          f'{sum(j["actual_steps"] for j in jobs):,} total environment steps', flush=True)
    if args.dry_run:
        for i, job in enumerate(jobs):
            print(f'CUDA_VISIBLE_DEVICES={slots[i % len(slots)]} {shlex.join(job["command"])}')
        return 0

    # Validate every GPU before launching any expensive jobs; never silently fall back to CPU.
    if not args.cpu:
        for gpu in slots:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            check = subprocess.run([sys.executable, '-c',
                'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; '
                'print(torch.cuda.get_device_name(0)); torch.zeros(1, device="cuda")'], env=env)
            if check.returncode:
                parser.error(f'GPU {gpu} failed the CUDA preflight')

    output = ROOT / 'results_compare/suites' / batch
    output.mkdir(parents=True, exist_ok=False)
    def git(*command):
        return subprocess.check_output(['git', *command], cwd=ROOT, text=True)
    manifest = dict(created=stamp(), suite=args.suite, slots=slots,
                    git_commit=git('rev-parse', 'HEAD').strip(),
                    git_status=git('status', '--short'), jobs=jobs)
    (output / 'tracked_changes.patch').write_text(git('diff', 'HEAD'))
    (output / 'launcher.py').write_bytes(Path(__file__).read_bytes())
    subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=(output / 'packages.txt').open('w'), check=True)
    manifest_path = output / 'manifest.json'
    def save():
        temporary = output / 'manifest.tmp'
        temporary.write_text(json.dumps(manifest, indent=2) + '\n')
        temporary.replace(manifest_path)
    save()
    print(f'Logs and status: {output}', flush=True)
    active = {}
    interrupted = False
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        pending = list(jobs)
        while pending or active:
            for slot in slots:
                if pending and slot not in active:
                    job = pending.pop(0)
                    log = (output / f'{job["name"]}.log').open('w')
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if args.cpu else slot,
                               PYTHONUNBUFFERED='1')
                    try:
                        process = subprocess.Popen(job['command'], cwd=ROOT, env=env,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    except BaseException:
                        log.close()
                        raise
                    job.update(status='running', device=slot, started=stamp(), pid=process.pid)
                    active[slot] = (process, log, job)
                    print(f'Started {job["name"]} on {slot}', flush=True)
                    save()
            for slot, (process, log, job) in list(active.items()):
                code = process.poll()
                if code is not None:
                    log.close()
                    job.update(status='completed' if code == 0 else 'failed', exit_code=code, finished=stamp())
                    del active[slot]
                    save()
                    print(f'{job["status"]}: {job["name"]} (exit {code})', flush=True)
            if active:
                time.sleep(1)
    except KeyboardInterrupt:
        interrupted = True
        print('Stopping active training process groups...', flush=True)
    finally:
        # Include rollout subprocesses in cancellation, not only the training parent.
        for process, log, job in active.values():
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process, log, job in active.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            log.close()
            job.update(status='interrupted', exit_code=process.returncode, finished=stamp())
        for job in jobs:
            if job['status'] == 'pending':
                job['status'] = 'not_started'
        save()
    return 130 if interrupted else int(any(j['status'] != 'completed' for j in jobs))


if __name__ == '__main__':
    sys.exit(main())
