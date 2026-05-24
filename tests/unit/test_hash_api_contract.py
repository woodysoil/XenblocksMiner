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
        "input_ms",
        "keygen_ms",
        "first_block_ms",
        "compute_ms",
        "finalize_ms",
        "total_ms",
    ]:
        assert field in types
        assert field in json_impl
        assert field in docs
    assert "timings" in json_impl
    assert "toJson(result.timings)" in json_impl
    assert "`timings`" in docs


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


def test_hash_api_cli_dispatches_cuda_backend_in_full_build():
    content = read("src/hashapi/HashApiCli.cpp")
    assert 'request.backend == "cuda"' in content
    assert "validateRequest(request)" in content
    assert "CudaHashBackend" in content
    assert "makeReusableBackend" in content
    assert "backend->runBatch(request)" in content
    assert "--difficulty-sequence" in content
    assert "parseDifficultySequence" in content
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
    assert "std::uniform_int_distribution<size_t> distribution" in content


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
