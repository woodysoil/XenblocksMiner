#!/usr/bin/env python3
"""Generate large test dataset for pagination testing."""

import argparse
import asyncio
import random
import time
import uuid
import sys

sys.path.insert(0, "/home/woody/XenblocksMiner")

GPU_PROFILES = [
    {"name": "NVIDIA GeForce RTX 4090", "memory_gb": 24, "hashrate_base": 350},
    {"name": "NVIDIA GeForce RTX 4080", "memory_gb": 16, "hashrate_base": 280},
    {"name": "NVIDIA GeForce RTX 4070 Ti", "memory_gb": 12, "hashrate_base": 200},
    {"name": "NVIDIA GeForce RTX 3090", "memory_gb": 24, "hashrate_base": 180},
    {"name": "NVIDIA GeForce RTX 3080 Ti", "memory_gb": 12, "hashrate_base": 150},
    {"name": "NVIDIA GeForce RTX 3080", "memory_gb": 10, "hashrate_base": 130},
    {"name": "NVIDIA GeForce RTX 3070", "memory_gb": 8, "hashrate_base": 100},
    {"name": "NVIDIA GeForce RTX 3060 Ti", "memory_gb": 8, "hashrate_base": 85},
    {"name": "AMD Radeon RX 7900 XTX", "memory_gb": 24, "hashrate_base": 200},
    {"name": "AMD Radeon RX 7900 XT", "memory_gb": 20, "hashrate_base": 170},
    {"name": "AMD Radeon RX 6900 XT", "memory_gb": 16, "hashrate_base": 140},
    {"name": "NVIDIA A100", "memory_gb": 80, "hashrate_base": 500},
    {"name": "NVIDIA H100", "memory_gb": 80, "hashrate_base": 800},
    {"name": "NVIDIA L40", "memory_gb": 48, "hashrate_base": 400},
]


async def generate_workers(storage, count: int, addresses: list[str]):
    """Generate fake workers using raw SQL for speed."""
    print(f"Generating {count} workers...")
    now = time.time()
    batch = []
    for i in range(count):
        gpu = random.choice(GPU_PROFILES)
        gpu_count = random.choice([1, 1, 1, 2, 2, 4, 8])
        gpus = [{"index": j, "name": gpu["name"], "memory_gb": gpu["memory_gb"]} for j in range(gpu_count)]
        gpus_json = __import__("json").dumps(gpus)

        batch.append((
            f"test-worker-{i:06d}",  # worker_id
            random.choice(addresses),  # eth_address
            gpu_count,  # gpu_count
            gpu["memory_gb"] * gpu_count,  # total_memory_gb
            gpus_json,  # gpus_json
            "2.0.0",  # version
            random.choice(["AVAILABLE", "SELF_MINING", "LEASED"]),  # state
            gpu["hashrate_base"] * gpu_count * random.uniform(0.8, 1.2),  # hashrate
            gpu_count,  # active_gpus
            now - random.randint(0, 300),  # last_heartbeat
            now,  # registered_at
            now,  # last_online_at
            random.randint(0, 500),  # self_blocks_found
            round(random.uniform(0.1, 2.0), 4),  # price_per_min
            300,  # min_duration_sec
            7200,  # max_duration_sec
            random.randint(3600, 86400 * 30),  # total_online_sec
        ))

        if len(batch) >= 500:
            await storage._db.executemany(
                "INSERT OR REPLACE INTO workers (worker_id, eth_address, gpu_count, total_memory_gb, gpus_json, "
                "version, state, hashrate, active_gpus, last_heartbeat, registered_at, last_online_at, "
                "self_blocks_found, price_per_min, min_duration_sec, max_duration_sec, total_online_sec) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                batch,
            )
            await storage._db.commit()
            batch = []
            if (i + 1) % 1000 == 0:
                print(f"  Created {i + 1} workers...")

    if batch:
        await storage._db.executemany(
            "INSERT OR REPLACE INTO workers (worker_id, eth_address, gpu_count, total_memory_gb, gpus_json, "
            "version, state, hashrate, active_gpus, last_heartbeat, registered_at, last_online_at, "
            "self_blocks_found, price_per_min, min_duration_sec, max_duration_sec, total_online_sec) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            batch,
        )
        await storage._db.commit()
    print(f"Created {count} workers")


async def generate_blocks(storage, count: int, worker_ids: list[str]):
    """Generate fake blocks using raw SQL for speed."""
    print(f"Generating {count} blocks...")
    now = time.time()
    batch = []
    for i in range(count):
        worker_id = random.choice(worker_ids)
        lease_id = "" if random.random() > 0.3 else f"lease-{uuid.uuid4().hex[:8]}"
        block_hash = f"0000{uuid.uuid4().hex[:56]}"
        key = uuid.uuid4().hex[:16]
        timestamp = now - random.randint(0, 86400 * 7)
        batch.append((lease_id, worker_id, block_hash, key, "", 0, "0.0", 1, 0, None, timestamp))

        if len(batch) >= 1000:
            await storage._db.executemany(
                "INSERT INTO blocks (lease_id, worker_id, block_hash, key, account, attempts, hashrate, prefix_valid, chain_verified, chain_block_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                batch,
            )
            await storage._db.commit()
            batch = []
            if (i + 1) % 10000 == 0:
                print(f"  Created {i + 1} blocks...")

    if batch:
        await storage._db.executemany(
            "INSERT INTO blocks (lease_id, worker_id, block_hash, key, account, attempts, hashrate, prefix_valid, chain_verified, chain_block_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            batch,
        )
        await storage._db.commit()
    print(f"Created {count} blocks")


async def generate_accounts(storage, consumer_ids: list[str]):
    """Generate consumer accounts."""
    print(f"Generating {len(consumer_ids)} consumer accounts...")
    now = time.time()
    batch = []
    for cid in consumer_ids:
        batch.append((cid, "consumer", f"0x{uuid.uuid4().hex[:40]}", 1000.0, now, now))

    await storage._db.executemany(
        "INSERT OR IGNORE INTO accounts (account_id, role, eth_address, balance, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        batch,
    )
    await storage._db.commit()
    print(f"Created {len(consumer_ids)} accounts")


async def generate_leases(storage, count: int, worker_ids: list[str], consumer_ids: list[str]):
    """Generate fake leases using raw SQL for speed."""
    print(f"Generating {count} leases...")
    now = time.time()
    batch = []
    for i in range(count):
        lease_id = f"lease-{uuid.uuid4().hex[:12]}"
        worker_id = random.choice(worker_ids)
        consumer_id = random.choice(consumer_ids)
        consumer_address = f"0x{uuid.uuid4().hex[:40]}"
        prefix = "0000"
        duration = random.randint(300, 7200)
        price_per_sec = round(random.uniform(0.001, 0.03), 6)
        state = random.choice(["completed", "completed", "completed", "active", "cancelled"])
        blocks_found = random.randint(0, 20) if state == "completed" else 0
        created = now - random.randint(0, 86400 * 30)
        ended = created + duration if state in ("completed", "cancelled") else None

        batch.append((lease_id, worker_id, consumer_id, consumer_address, prefix, duration, price_per_sec, state, created, ended, blocks_found, 0.0, 0))

        if len(batch) >= 1000:
            await storage._db.executemany(
                "INSERT INTO leases (lease_id, worker_id, consumer_id, consumer_address, prefix, duration_sec, price_per_sec, state, created_at, ended_at, blocks_found, total_hashrate_samples, hashrate_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                batch,
            )
            await storage._db.commit()
            batch = []
            if (i + 1) % 1000 == 0:
                print(f"  Created {i + 1} leases...")

    if batch:
        await storage._db.executemany(
            "INSERT INTO leases (lease_id, worker_id, consumer_id, consumer_address, prefix, duration_sec, price_per_sec, state, created_at, ended_at, blocks_found, total_hashrate_samples, hashrate_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            batch,
        )
        await storage._db.commit()
    print(f"Created {count} leases")


async def main():
    parser = argparse.ArgumentParser(description="Generate test data for pagination")
    parser.add_argument("--workers", type=int, default=1000, help="Number of workers")
    parser.add_argument("--blocks", type=int, default=50000, help="Number of blocks")
    parser.add_argument("--leases", type=int, default=5000, help="Number of leases")
    parser.add_argument("--addresses", type=int, default=100, help="Unique wallet addresses")
    parser.add_argument("--db", type=str, default="data/server.db", help="Database path")
    args = parser.parse_args()

    from server.storage import StorageManager

    print(f"Connecting to database: {args.db}")
    storage = StorageManager(args.db)
    await storage.initialize()

    # Generate addresses
    addresses = [f"0x{uuid.uuid4().hex[:40]}" for _ in range(args.addresses)]
    print(f"Using {len(addresses)} unique wallet addresses")

    # Generate workers
    await generate_workers(storage, args.workers, addresses)

    # Get worker IDs
    all_workers = await storage.workers.list_all()
    worker_ids = [w["worker_id"] for w in all_workers]
    print(f"Total workers in DB: {len(worker_ids)}")

    # Generate blocks
    await generate_blocks(storage, args.blocks, worker_ids)

    # Generate leases
    consumer_ids = [f"consumer-{uuid.uuid4().hex[:8]}" for _ in range(50)]
    await generate_accounts(storage, consumer_ids)
    await generate_leases(storage, args.leases, worker_ids, consumer_ids)

    # Summary
    total_workers = await storage.workers.count()
    total_blocks = await storage.blocks.count()
    total_leases = await storage.leases.count()
    print(f"\n=== Summary ===")
    print(f"Workers: {total_workers:,}")
    print(f"Blocks:  {total_blocks:,}")
    print(f"Leases:  {total_leases:,}")


if __name__ == "__main__":
    asyncio.run(main())
