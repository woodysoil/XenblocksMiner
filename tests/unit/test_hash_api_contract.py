"""Static contract checks for the C++ Hash API boundary."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_hash_api_type_contract_exists():
    content = read("src/hashapi/HashApiTypes.h")
    for token in [
        "struct HashApiRequest",
        "struct HashApiMatch",
        "struct HashApiTimings",
        "struct HashApiResult",
        "class IHashBackend",
        "virtual HashApiResult runBatch",
    ]:
        assert token in content


def test_hash_api_request_fields_exist():
    content = read("src/hashapi/HashApiTypes.h")
    for field in [
        "request_id",
        "algorithm",
        "backend",
        "salt_hex",
        "key_prefix",
        "target_pattern",
        "difficulty",
        "batch_size",
        "device_id",
        "allow_xuni",
        "detailed_timings",
        "first_block_workers",
        "first_block_dynamic_chunk_size",
        "first_block_dynamic_chunk_auto",
        "gpu_first_blocks",
    ]:
        assert field in content


def test_hash_api_validation_rules_are_implemented():
    content = read("src/hashapi/HashApiValidation.cpp")
    for rule in [
        "unsupported algorithm",
        "unsupported backend",
        "salt_hex is required",
        "key_prefix cannot exceed 64 hex characters",
        "key must contain exactly 64 hex characters",
        "target_pattern is required",
        "difficulty must be greater than zero",
        "batch_size must be greater than zero",
        "device_id must be non-negative",
        "gpu_first_blocks requires backend=cuda",
    ]:
        assert rule in content


def test_cpu_hash_api_backend_declares_argon2_minimum_difficulty():
    content = read("src/hashapi/CpuHashBackend.cpp")
    assert "kMinArgon2CpuDifficulty = 8" in content
    assert "cpu/reference difficulty must be at least 8" in content


def test_hash_api_docs_reference_cli_and_boundaries():
    content = read("docs/hash-api.md")
    lower_content = content.lower()
    for token in [
        "hash-one",
        "hash-batch",
        "hash-benchmark",
        "The Hash API does not own",
        "Validation Rules",
    ]:
        assert token in content
    assert "stub backend" in lower_content


def test_hash_api_smoke_preset_exists():
    content = read("CMakePresets.json")
    assert "hashapi-cli-smoke-mingw" in content
    assert "XENBLOCKS_HASHAPI_STUB_BACKEND" in content


def test_cuda_release_benchmark_presets_exist():
    content = read("CMakePresets.json")
    docs = read("doc/BUILD_INSTRUCTIONS.md")

    for token in [
        "cuda-release-vcpkg-modern",
        "cuda-release-vcpkg-sm86",
        "cuda-release-vcpkg-sm89",
        "cuda-release-vcpkg-sm90",
        '"CMAKE_BUILD_TYPE": "Release"',
        '"CMAKE_CUDA_ARCHITECTURES": "75;80;86;89;90"',
    ]:
        assert token in content

    for token in [
        "repeatable Hash API/CUDA benchmark runs",
        "cuda-release-vcpkg-modern",
        "cuda-release-vcpkg-sm86",
        "Do not compare benchmark results from a Debug build",
        "CMAKE_CUDA_ARCHITECTURES",
    ]:
        assert token in docs


def test_hash_api_json_uses_standard_library_only():
    content = read("src/hashapi/HashApiJson.h")
    assert "nlohmann/json.hpp" not in content
    assert "std::string toJson" in content


def test_hash_api_result_exposes_machine_readable_timings():
    types = read("src/hashapi/HashApiTypes.h")
    json_impl = read("src/hashapi/HashApiJson.cpp")
    docs = read("docs/hash-api.md")

    for field in [
        "validation_ms",
        "setup_ms",
        "setup_normalize_cpu_ms",
        "setup_activate_cpu_ms",
        "setup_device_info_cpu_ms",
        "setup_params_cpu_ms",
        "setup_backend_init_cpu_ms",
        "input_ms",
        "keygen_ms",
        "first_block_ms",
        "first_block_initial_hash_cpu_ms",
        "first_block_digest_cpu_ms",
        "first_block_max_worker_ms",
        "first_block_thread_launch_ms",
        "first_block_max_worker_start_ms",
        "first_block_worker_start_span_ms",
        "first_block_max_worker_finish_ms",
        "first_block_worker_finish_span_ms",
        "compute_ms",
        "kernel_ms",
        "host_to_device_ms",
        "gpu_first_block_ms",
        "device_to_host_ms",
        "finalize_ms",
        "finalize_hash_ms",
        "argon2_finalize_ms",
        "base64_ms",
        "match_ms",
        "total_ms",
    ]:
        assert field in types
        assert field in json_impl
        assert field in docs
    assert "timings" in json_impl
    assert "toJson(result.timings)" in json_impl
    assert "`timings`" in docs


def test_hash_api_result_exposes_first_block_scheduling_metadata():
    types = read("src/hashapi/HashApiTypes.h")
    json_impl = read("src/hashapi/HashApiJson.cpp")
    cuda_impl = read("src/hashapi/CudaHashBackend.cpp")
    docs = read("docs/hash-api.md")

    for field in [
        "first_block_dynamic_chunk_size",
        "first_block_dynamic_chunk_auto",
        "first_block_worker_count",
        "first_block_chunk_size",
        "first_block_dynamic_chunk_size_min",
        "first_block_dynamic_chunk_size_max",
        "first_block_chunk_size_min",
        "first_block_chunk_size_max",
    ]:
        assert field in types
        assert field in json_impl
        assert field in docs

    assert "firstBlockWorkerCount(attempts, request.first_block_workers)" in cuda_impl
    assert "firstBlockSelectedChunkSize(" in cuda_impl
    assert "update_first_block_ranges(current)" in read("src/hashapi/HashApiCli.cpp")
    assert "request.first_block_dynamic_chunk_size" in cuda_impl
    assert "recommendedFirstBlockDynamicChunkSize" in cuda_impl
    assert "request.first_block_dynamic_chunk_auto" in cuda_impl
    assert "request.difficulty == 1" in cuda_impl
    assert "request.difficulty == 8" in cuda_impl
    assert "attempts >= 2048 ? 16 : 32" in cuda_impl
    assert "request.difficulty == 64" in cuda_impl
    assert "attempts <= 2048 ? 16 : 0" in cuda_impl
    assert "next_dynamic_index.fetch_add(chunk_size, std::memory_order_relaxed)" in cuda_impl


def test_hash_api_exposes_gpu_first_block_experiment_flag():
    types = read("src/hashapi/HashApiTypes.h")
    json_impl = read("src/hashapi/HashApiJson.cpp")
    cli_impl = read("src/hashapi/HashApiCli.cpp")
    compute = read("src/ComputeBackend.h")
    cuda = read("src/hashapi/CudaHashBackend.cpp")
    runner = read("src/kernelrunner.cu")

    assert "bool gpu_first_blocks = false" in types
    assert "gpu_first_blocks" in json_impl
    assert "--gpu-first-blocks" in cli_impl
    assert "prepareInputBlocksOnDevice" in compute
    assert "getLastGpuFirstBlockMs" in compute
    assert "request.gpu_first_blocks" in cuda
    assert "prepareInputBlocksOnDevice(password_storage_" in cuda
    assert "argon2_first_blocks_kernel" in runner
    assert "getLastGpuFirstBlockMs" in runner


def test_hash_api_result_exposes_batch_size_range_metadata():
    types = read("src/hashapi/HashApiTypes.h")
    json_impl = read("src/hashapi/HashApiJson.cpp")
    cli_impl = read("src/hashapi/HashApiCli.cpp")
    docs = read("docs/hash-api.md")

    for field in [
        "batch_size_min",
        "batch_size_max",
    ]:
        assert field in types
        assert field in json_impl
        assert field in docs

    assert "--batch-size-sequence" in cli_impl
    assert "parseBatchSizeSequence" in cli_impl
    assert "update_batch_size_ranges(current.batch_size)" in cli_impl
    assert "difficulty sequence and batch-size sequence lengths must match" in cli_impl
    assert "--batch-size-sequence" in docs


def test_hash_api_base64_encoder_avoids_incremental_string_appends():
    content = read("src/hashapi/HashApiEncoding.cpp")
    header = read("src/hashapi/HashApiEncoding.h")
    assert "base64EncodedLength" in header
    assert "base64EncodeInto" in header
    assert "encoded.reserve(base64EncodedLength(in_len))" in content
    assert "base64EncodeInto(encoded, bytes_to_encode, in_len)" in content
    assert "encoded.push_back" in content
    assert "ret +=" not in content


def test_cuda_hash_api_backend_exists():
    header = read("src/hashapi/CudaHashBackend.h")
    implementation = read("src/hashapi/CudaHashBackend.cpp")
    cmake = read("CMakeLists.txt")

    assert "class CudaHashBackend" in header
    assert "public IHashBackend" in header
    assert "HashApiResult CudaHashBackend::runBatch" in implementation
    assert "ComputeBackend" in implementation
    assert "appendMatches" in implementation
    assert "src/hashapi/CudaHashBackend.cpp" in cmake


def test_cuda_hash_api_reuses_initialization_by_segment_blocks():
    header = read("src/hashapi/CudaHashBackend.h")
    implementation = read("src/hashapi/CudaHashBackend.cpp")

    assert "initialized_segment_blocks_" in header
    assert "initialized_difficulty_" not in header
    assert "initialized_difficulty_" not in implementation
    assert "initialized_segment_blocks_ == segment_blocks" in implementation


def test_cuda_backend_reuses_runner_when_allocation_covers_segment_blocks():
    backend_impl = read("src/CudaBackend.cpp")
    runner_header = read("src/kernelrunner.h")
    runner_impl = read("src/kernelrunner.cu")

    assert "runner_->canReuse(type, version, passes, lanes, segmentBlocks, batchSize)" in backend_impl
    assert "runner_->reconfigure(type, version, passes, lanes, segmentBlocks, batchSize)" in backend_impl
    assert "allocatedSegmentBlocks" in runner_header
    assert "segmentBlocks_ <= allocatedSegmentBlocks" in runner_impl
    assert "allocatedSegmentBlocks = segmentBlocks" in runner_impl


def test_hash_api_cli_dispatches_cuda_backend_in_full_build():
    content = read("src/hashapi/HashApiCli.cpp")
    assert 'request.backend == "cuda"' in content
    assert "validateRequest(request)" in content
    assert "CudaHashBackend" in content
    assert "makeReusableBackend" in content
    assert "selectAutomaticCudaBatchSize" in content
    assert "selectCudaBatchSizeForDifficultySequence" in content
    assert "--auto-batch-size" in content
    assert "backend->runBatch(request)" in content
    assert "--difficulty-sequence" in content
    assert "--batch-size-sequence" in content
    assert "parseDifficultySequence" in content
    assert "parseBatchSizeSequence" in content
    assert "aggregate.hash = current.hash" in content
    assert "aggregate.hash.clear()" in content
    assert "aggregate.first_block_dynamic_chunk_size = current.first_block_dynamic_chunk_size" in content
    assert "aggregate.first_block_dynamic_chunk_auto = current.first_block_dynamic_chunk_auto" in content
    assert "aggregate.first_block_worker_count = current.first_block_worker_count" in content
    assert "aggregate.first_block_chunk_size = current.first_block_chunk_size" in content
    assert "target.first_block_max_worker_ms += source.first_block_max_worker_ms" in content
    assert "target.first_block_thread_launch_ms += source.first_block_thread_launch_ms" in content
    assert "target.first_block_worker_finish_span_ms += source.first_block_worker_finish_span_ms" in content
    assert "--detailed-timings" in content
    assert "--first-block-dynamic-chunk-size" in content
    assert "--first-block-dynamic-chunk-auto" in content
    assert "ex.what()" in content
    assert "cuda backend is not available in this build" in content


def test_mine_unit_routes_batch_compute_through_hash_api():
    header = read("src/MineUnit.h")
    implementation = read("src/MineUnit.cpp")

    assert "hashapi::CudaHashBackend hashBackend_" in header
    assert "hashapi::HashApiResult batchCompute" in header
    assert "hashBackend_.runBatch(request)" in implementation
    assert "request.allow_xuni = is_within_five_minutes_of_hour()" in implementation
    assert "request.first_block_dynamic_chunk_auto = true" in implementation
    assert "submitMatches" in implementation
    assert "std::vector<HashItem>" not in header


def test_cuda_batch_size_tuning_helper_exists():
    header = read("src/hashapi/HashApiTuning.h")
    implementation = read("src/hashapi/HashApiTuning.cpp")
    cmake = read("CMakeLists.txt")

    for token in [
        "struct CudaBatchSizeDecision",
        "estimateCudaMemoryBatchLimit",
        "recommendedCudaBatchSize",
        "recommendedCudaBatchSizeForDifficultySequence",
        "selectCudaBatchSize",
        "selectCudaBatchSizeForDifficultySequence",
    ]:
        assert token in header

    for token in [
        "estimateCudaMemoryBatchLimit",
        "recommendedCudaBatchSize",
        "recommendedCudaBatchSizeForDifficultySequence",
        "selectCudaBatchSize",
        "selectCudaBatchSizeForDifficultySequence",
    ]:
        assert token in implementation

    assert "kCudaBatchMemoryReserveBytes" in header
    assert "difficulty <= 1" in implementation
    assert "return 2048" in implementation
    assert "difficulty <= 8" in implementation
    assert "return 4096" in implementation
    assert "difficulty <= 64" in implementation
    assert "return 3072" in implementation
    assert "std::min(selected, recommended)" in implementation
    assert "std::max_element(difficulties.begin(), difficulties.end())" in implementation
    assert "explicit_max_batch_size > 0" in implementation
    assert "src/hashapi/HashApiTuning.cpp" in cmake


def test_blake2b_copy_selftest_target_is_available():
    cmake = read("CMakeLists.txt")
    source = read("tests/cpp/blake2b_copy_selftest.cpp")

    assert "XENBLOCKS_BUILD_HASH_DIAGNOSTICS" in cmake
    assert "blake2b-copy-selftest" in cmake
    assert "tests/cpp/blake2b_copy_selftest.cpp" in cmake
    assert "copied_states_are_independent" in source


def test_argon2_finalize_benchmark_target_is_available():
    cmake = read("CMakeLists.txt")
    source = read("tests/cpp/argon2_finalize_benchmark.cpp")

    assert "argon2-finalize-benchmark" in cmake
    assert "tests/cpp/argon2_finalize_benchmark.cpp" in cmake
    assert "argon2id-xen-finalize" in source
    assert "ns_per_finalize" in source
    assert "sample_hash" in source
    assert "known_sample_ok" in source
    assert "kExpectedDefaultChecksum" in source


def test_mine_unit_uses_hash_api_batch_size_tuning_without_overriding_manual_limit():
    implementation = read("src/MineUnit.cpp")

    assert '#include "hashapi/HashApiTuning.h"' in implementation
    assert "hashapi::selectCudaBatchSize" in implementation
    assert "globalMaxBatchSize" in implementation
    assert "selected_batch_size == 0" in implementation
    assert "batchSize = batchDecision.selected_batch_size" in implementation
    assert "request.gpu_first_blocks = true" in implementation


def test_hash_api_benchmark_runner_exists():
    content = read("scripts/hash_api_benchmark.py")
    docs = read("docs/hash-api.md")

    assert "xenblocks.hashapi.benchmark.v1" in content
    assert "hash-benchmark" in content
    assert "difficulty_sequence" in content
    assert "difficulty-sequence" in content
    assert "sequence_auto_batch_size" in content
    assert "--sequence-auto-batch-size" in content
    assert "sequence_first_block_dynamic_chunk_auto" in content
    assert "--sequence-first-block-dynamic-chunk-auto" in content
    assert "capture_output=True" in content
    assert "nvidia-smi" in content
    assert "nvcc" in content
    assert "summary" in content
    assert "scripts/hash_api_benchmark.py" in docs
    assert "<miner-binary>" in docs
    assert "--difficulty-sequence" in docs
    assert "--batch-size-sequence" in docs


def test_random_key_generator_avoids_per_key_stream_allocation():
    content = read("src/RandomHexKeyGenerator.h")
    assert "std::stringstream" not in content
    assert "key.reserve(total_length)" in content
    assert "std::uniform_int_distribution" not in content
    assert "std::uint32_t random_bits = generator()" in content


def test_hash_api_matching_avoids_regex_in_hot_path():
    content = read("src/hashapi/HashApiMatching.cpp")
    assert "std::regex" not in content
    assert "std::regex_search" not in content
    assert 'hash.find(kXuniPrefix)' in content


def test_local_hash_service_is_separate_from_marketplace_server():
    service = read("server/hash_api/app.py")
    platform_server = read("server/server.py")
    docs = read("docs/hash-api.md")

    assert "/hash/v1/health" in service
    assert "/hash/v1/backends" in service
    assert "/hash/v1/validate" in service
    assert "/hash/v1/hash-one" in service
    assert "/hash/v1/batch" in service
    assert "/hash/v1/benchmark" in service
    assert "server.hash_api" not in platform_server
    assert "separate FastAPI app" in docs
