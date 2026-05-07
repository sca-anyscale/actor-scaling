"""Benchmark actor scheduling at scale.

Usage:
    # Smoke test (local or small cluster)
    python bench_actor_scheduling.py --num-actors 20 --num-cpus-per-actor 1

    # Large scale with fake GPU (simulate GPU actors)
    python bench_actor_scheduling.py --num-actors 16000 --resource fake_GPU=1

    # With placement groups
    python bench_actor_scheduling.py --num-actors 16000 --resource fake_GPU=1 --use-pg
"""
import argparse
import os
import time

import ray
from profiling.coordinator import Profiling

BENCHMARK = 'sca-actor-scaling'

@ray.remote
class DummyActor:
    def ready(self):
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-actors", type=int, default=20)
    parser.add_argument("--num-cpus-per-actor", type=float, default=0)
    parser.add_argument(
        "--resource",
        type=str,
        default=None,
        help="Custom resource requirement, e.g. fake_GPU=1",
    )
    parser.add_argument("--use-pg", action="store_true", help="Schedule actors inside placement groups")
    parser.add_argument("--profile", action='store_true')
    args = parser.parse_args()

    ray.init()
    if args.profile:
        job_id = os.environ.get("ANYSCALE_JOB_ID", "unknown")
        if job_id == 'unknown':
            job_id = os.environ.get("ANYSCALE_WORKSPACE_ID", "unknown")

        profiling = Profiling(
            outdir=f"/mnt/shared_storage/{BENCHMARK}/{job_id}",
            num_gpu_nodes=0,
        )

        profiling.start()

    nodes = ray.nodes()
    alive_nodes = [n for n in nodes if n["Alive"]]
    print(f"Cluster: {len(alive_nodes)} alive nodes")
    for n in alive_nodes[:3]:
        res = n.get("Resources", {})
        print(f"  Node {n['NodeID'][:8]}... CPU={res.get('CPU', 0)} fake_GPU={res.get('fake_GPU', 0)}")
    if len(alive_nodes) > 3:
        print(f"  ... and {len(alive_nodes) - 3} more")

    num_actors = args.num_actors
    options = {"num_cpus": args.num_cpus_per_actor}
    if args.resource:
        key, val = args.resource.split("=")
        options["resources"] = {key: float(val)}

    actor_cls = DummyActor.options(**options)

    print(f"\nCreating {num_actors} actors (options={options})...")
    t0 = time.time()
    actors = [actor_cls.remote() for _ in range(num_actors)]
    t_create = time.time()
    print(f"  ray.remote() calls done in {t_create - t0:.2f}s")

    refs = [a.ready.remote() for a in actors]
    ray.get(refs)
    t_ready = time.time()

    print(f"\nResults:")
    print(f"  Submission:              {t_create - t0:.2f}s")
    print(f"  Scheduling + startup:    {t_ready - t_create:.2f}s")
    print(f"  Total:                   {t_ready - t0:.2f}s")

    if args.profile:
        profiling.stop(s3_prefix=f"{BENCHMARK}/{job_id}")
    ray.shutdown()


if __name__ == "__main__":
    main()
