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
        "first_block_worker_count",
        "first_block_chunk_size",
    ]:
        assert field in types
        assert field in json_impl
        assert field in docs

    assert "firstBlockWorkerCount(attempts, request.first_block_workers)" in cuda_impl
    assert "firstBlockChunkSize(attempts, result.first_block_worker_count)" in cuda_impl


def test_hash_api_base64_encoder_avoids_incremental_string_appends():
    content = read("src/hashapi/HashApiEncoding.cpp")
    assert "encoded.reserve(((in_len + 2) / 3) * 4)" in content
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


def test_hash_api_cli_dispatches_cuda_backend_in_full_build():
    content = read("src/hashapi/HashApiCli.cpp")
    assert 'request.backend == "cuda"' in content
    assert "validateRequest(request)" in content
    assert "CudaHashBackend" in content
    assert "makeReusableBackend" in content
    assert "backend->runBatch(request)" in content
    assert "--difficulty-sequence" in content
    assert "parseDifficultySequence" in content
    assert "aggregate.hash = current.hash" in content
    assert "aggregate.hash.clear()" in content
    assert "aggregate.first_block_worker_count = current.first_block_worker_count" in content
    assert "aggregate.first_block_chunk_size = current.first_block_chunk_size" in content
    assert "target.first_block_max_worker_ms += source.first_block_max_worker_ms" in content
    assert "target.first_block_thread_launch_ms += source.first_block_thread_launch_ms" in content
    assert "target.first_block_worker_finish_span_ms += source.first_block_worker_finish_span_ms" in content
    assert "--detailed-timings" in content
    assert "ex.what()" in content
    assert "cuda backend is not available in this build" in content


def test_mine_unit_routes_batch_compute_through_hash_api():
    header = read("src/MineUnit.h")
    implementation = read("src/MineUnit.cpp")

    assert "hashapi::CudaHashBackend hashBackend_" in header
    assert "hashapi::HashApiResult batchCompute" in header
    assert "hashBackend_.runBatch(request)" in implementation
    assert "request.allow_xuni = is_within_five_minutes_of_hour()" in implementation
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
        "selectCudaBatchSize",
    ]:
        assert token in header

    for token in [
        "estimateCudaMemoryBatchLimit",
        "recommendedCudaBatchSize",
        "selectCudaBatchSize",
    ]:
        assert token in implementation

    assert "kCudaBatchMemoryReserveBytes" in header
    assert "difficulty <= 1" in implementation
    assert "return 512" in implementation
    assert "difficulty <= 8" in implementation
    assert "return 2048" in implementation
    assert "difficulty <= 64" in implementation
    assert "return 512" in implementation
    assert "explicit_max_batch_size > 0" in implementation
    assert "src/hashapi/HashApiTuning.cpp" in cmake


def test_mine_unit_uses_hash_api_batch_size_tuning_without_overriding_manual_limit():
    implementation = read("src/MineUnit.cpp")

    assert '#include "hashapi/HashApiTuning.h"' in implementation
    assert "hashapi::selectCudaBatchSize" in implementation
    assert "globalMaxBatchSize" in implementation
    assert "selected_batch_size == 0" in implementation
    assert "batchSize = batchDecision.selected_batch_size" in implementation


def test_hash_api_benchmark_runner_exists():
    content = read("scripts/hash_api_benchmark.py")
    docs = read("docs/hash-api.md")

    assert "xenblocks.hashapi.benchmark.v1" in content
    assert "hash-benchmark" in content
    assert "difficulty_sequence" in content
    assert "difficulty-sequence" in content
    assert "capture_output=True" in content
    assert "nvidia-smi" in content
    assert "nvcc" in content
    assert "summary" in content
    assert "scripts/hash_api_benchmark.py" in docs
    assert "<miner-binary>" in docs
    assert "--difficulty-sequence" in docs


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
