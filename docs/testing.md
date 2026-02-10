# 测试指南

## 测试架构

项目采用 **unit + integration** 双层测试结构，全部基于 pytest。

```
tests/
├── unit/                    # 纯逻辑验证，无外部依赖
│   ├── conftest.py          # 共享 fixtures：ETH 地址、hex 前缀、搜索串
│   ├── mock_mqtt_broker.py  # 进程内 MQTT broker mock（支持 +/# 通配符）
│   ├── test_key_prefix.py   # RandomHexKeyGenerator 前缀注入
│   ├── test_mining_flow.py  # 挖矿流程（SHA-256/Argon2id、DevFee、模式切换）
│   ├── test_monitoring.py   # MonitoringService（内存 SQLite）
│   ├── test_mqtt_integration.py  # MockMQTTBroker pub/sub + 完整注册流
│   ├── test_security.py     # 前缀长度、ETH 地址、HMAC 签名、注入边界
│   └── test_state_machine.py     # 6 状态机（合法/非法转换、超时、回调）
└── integration/             # 组件协作 + REST/WebSocket 端点
    ├── conftest.py          # WorkerSimulator、PlatformSimulator、消息工厂
    ├── test_block_discovery.py   # block_found 消息结构与状态守卫
    ├── test_cpp_worker.py        # 真实 C++ 矿机 ↔ mock server（需 CUDA + 编译产物）
    ├── test_error_recovery.py    # 注册拒绝、pause/resume/shutdown、无效状态下的任务分配
    ├── test_lease_flow.py        # assign_task → LEASED → MINING → release → AVAILABLE
    ├── test_monitoring_api.py    # /api/monitoring/* REST 端点（FastAPI TestClient）
    ├── test_multi_worker.py      # 多 worker 注册/租约隔离/区块归属/心跳独立
    ├── test_settlement.py        # 结算数据完整性（区块计数、lease_id、消费者地址）
    ├── test_websocket.py         # WebSocket /ws/dashboard 快照推送与广播
    └── test_worker_registration.py  # 注册握手、GPU 信息、ACK/NACK、重注册
```

### 关键基础设施

- **`mock_mqtt_broker.py`** -- 进程内 MQTT broker，支持 `+`/`#` 通配符、retained 消息、JSON 序列化、`on_message` 回调。所有 unit 和 integration 测试复用此组件，无需外部 broker。
- **`integration/conftest.py`** -- 提供 `WorkerSimulator`（模拟 C++ 矿机 MQTT 行为）和 `PlatformSimulator`（模拟平台服务端），以及协议消息工厂函数（`make_register_msg`、`make_assign_task` 等）。

## 运行测试

### 全量测试（CI 默认）

```bash
python3 -m pytest tests/ -v --tb=short --ignore=tests/integration/test_cpp_worker.py
```

### 仅 unit

```bash
python3 -m pytest tests/unit/ -v
```

### 仅 integration（不含 C++ 测试）

```bash
python3 -m pytest tests/integration/ -v --ignore=tests/integration/test_cpp_worker.py
```

### C++ 矿机集成测试（需 CUDA GPU + 编译产物）

```bash
# 自动检测 build/ 下的二进制
python3 -m pytest tests/integration/test_cpp_worker.py -v --tb=short

# 手动指定二进制路径
MINER_BIN=/path/to/xenblocksMiner python3 -m pytest tests/integration/test_cpp_worker.py -v
```

无 CUDA 或未编译时自动 skip。

### 单文件

```bash
python3 -m pytest tests/unit/test_state_machine.py -v
```

## 各测试文件覆盖范围

### Unit 测试

| 文件 | 覆盖范围 |
|---|---|
| `test_key_prefix.py` | `RandomHexKeyGenerator` 的 Python 移植：key 长度恒为 64、hex 合法性、前缀注入（DevFee 8 字符 / 平台 16 字符）、大小写归一、边界（空前缀 / 满长 / 超长）、DevFee vs 平台优先级、salt 构造、随机性 |
| `test_mining_flow.py` | `SimulatedMiner` 端到端：自挖矿/平台模式前缀解析、简化搜索("ab")快速出块、模式切换（self→platform→self）、DevFee 轮转（计数器回绕 / 触发时机 / Eco）、Argon2id 集成（可选）、SubmitCallback 仿真 |
| `test_monitoring.py` | `MonitoringService`：fleet overview（在线/离线判定）、聚合统计（hashrate / GPU / 区块数）、hashrate 快照记录与查询、旧快照清理、worker 健康检测、recent blocks |
| `test_mqtt_integration.py` | `MockMQTTBroker`：topic 通配符匹配、connect/disconnect/pub/sub 基础操作、JSON payload、retained 消息、on_message 回调、完整注册流（register→ACK→assign→status→block_found）、心跳超时检测、多矿机隔离 |
| `test_security.py` | 平台前缀长度限制（1-16 合法，0 和 17+ 拒绝）、非 hex 字符拒绝、ETH 地址格式校验、MQTT payload 必填字段验证、HMAC-SHA256 签名/验签、key 注入边界（不超 HASH_LENGTH）、DevFee 前缀防伪 |
| `test_state_machine.py` | `PlatformStateMachine` 6 状态（IDLE→AVAILABLE→LEASED→MINING→COMPLETED→ERROR）：所有合法转换、所有非法转换（parametrize）、超时检测与重置、on_enter/on_exit 回调、force reset、转换图完整性、错误恢复路径（单次/连续/混合） |

### Integration 测试

| 文件 | 覆盖范围 |
|---|---|
| `test_worker_registration.py` | 注册消息 schema 合规（7 个必填字段）、GPU 数组结构、register_ack 接受/拒绝流、重注册（断连重连 / resume 触发）、边界（自定义 ETH 地址 / 重复 worker_id 覆写） |
| `test_lease_flow.py` | assign_task 消息结构（prefix 16 hex / duration / consumer_address）、租约分配与状态转换（AVAILABLE→LEASED→MINING）、release 流程（正确 lease_id / 空 id / 错误 id）、顺序多租约、前缀与时长字段验证 |
| `test_block_discovery.py` | block_found 消息结构（hashrate 为 string、attempts 为 int）、仅 MINING 状态可上报、key 前缀匹配 lease prefix、单租约多区块、跨租约区块独立追踪、account 字段与消费者地址一致 |
| `test_error_recovery.py` | 注册拒绝后恢复、pause→IDLE→resume→AVAILABLE、resume 触发重注册、挖矿中 pause 释放租约、shutdown 断连与 offline 状态上报、无效状态下 assign_task 被忽略、release 错误 lease_id 被忽略、lease 释放后重新接受任务 |
| `test_multi_worker.py` | 多 worker 独立注册、租约隔离（分配给 A 不影响 B）、release 隔离、不同消费者同时租用、区块归属正确（worker_id + prefix）、心跳独立追踪、control 命令隔离（pause/shutdown 单个 worker）、MQTT topic 命名空间隔离 |
| `test_settlement.py` | 租约完成后区块计数、lease_id 一致性、零区块租约正常完成、区块 worker_id / account / key prefix 完整性、时间戳有序、多租约结算数据独立归属、COMPLETED 与 AVAILABLE 状态流 |
| `test_monitoring_api.py` | `/api/monitoring/fleet`（空列表 / 在线离线 / 字段完整）、`/api/monitoring/stats`（聚合值 / 区块计数）、`/api/monitoring/hashrate-history`（快照查询 / worker 过滤）、`/api/monitoring/blocks/recent`（limit / 降序）、性能测试（100 worker < 500ms） |
| `test_websocket.py` | `/ws/dashboard` 连接即获 snapshot、snapshot 含 workers + recent_blocks、heartbeat/block 广播、多客户端同时接收、断连清理、断连后广播不报错 |
| `test_cpp_worker.py` | 真实 C++ 矿机二进制：注册验证（worker 列表 / state / GPU info / ETH 地址）、平台状态端点（:42069/platform/status、/stats）、完整租约流（rent→mining→block验证→stop→settlement→recovery）、心跳更新、control 命令稳定性。需 CUDA + 编译产物，否则自动 skip |

## 钱包认证手动测试

服务端使用 EIP-191 签名认证，流程：获取 nonce → 签名 → 验证 → 获取 JWT。

### 依赖

```bash
pip install eth-account requests
```

### 步骤

```python
from eth_account import Account
from eth_account.messages import encode_defunct
import requests

BASE = "http://localhost:8080"

# 1. 生成钱包
acct = Account.create()
addr = acct.address

# 2. 获取 nonce
r = requests.get(f"{BASE}/api/auth/nonce?address={addr}")
nonce = r.json()["nonce"]

# 3. 签名
msg = encode_defunct(text=f"Sign this message to authenticate with XenBlocks.\n\nNonce: {nonce}")
sig = acct.sign_message(msg).signature.hex()

# 4. 验证并获取 JWT
r = requests.post(f"{BASE}/api/auth/verify", json={
    "address": addr,
    "signature": f"0x{sig}",
    "nonce": nonce,
})
jwt_token = r.json()["token"]

# 5. 使用 JWT 调用受保护接口
r = requests.get(f"{BASE}/api/account/me",
                 headers={"Authorization": f"Bearer {jwt_token}"})
print(r.json())
```

关键参数：
- Nonce TTL：300 秒（5 分钟内完成签名）
- JWT TTL：86400 秒（24 小时）
- 签名消息模板：`Sign this message to authenticate with XenBlocks.\n\nNonce: {nonce}`
- 也支持 legacy API key 方式：`X-API-Key` header

## Mock Fleet 模拟器

`scripts/mock_fleet.py` 可启动 N 个模拟矿机连接 MQTT broker，用于压测和 dashboard 演示。

### 用法

```bash
python scripts/mock_fleet.py \
  --workers 10 \
  --broker localhost \
  --port 1883 \
  --block-interval 60 \
  --owner 0xYourEthAddress
```

### 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--workers` | 5 | 模拟 worker 数量 |
| `--broker` | localhost | MQTT broker 地址 |
| `--port` | 1883 | MQTT broker 端口 |
| `--block-interval` | 60 | 每个 worker 平均出块间隔（秒） |
| `--owner` | 随机 | 所有 worker 使用同一 ETH 地址（不指定则每个 worker 随机生成） |

### 行为

- 启动时所有 worker 连接 broker 并发送注册消息
- 每 30 秒发送心跳（hashrate 有随机波动）
- 按泊松分布发现区块（lambda = 1/block-interval）
- 每 5 分钟 10% 概率随机下线 30-120 秒后自动恢复
- 从 4 种 GPU 配置中随机选择（RTX 4090/3090/4080, A100）

## CI 测试流程

CI 配置位于 `.github/workflows/linux-build.yml`，在 `nvidia/cuda:11.8.0-devel-ubuntu22.04` 容器中运行。

### 流程

1. 安装系统依赖（git, cmake, ninja 等）
2. 恢复 vcpkg 缓存
3. CMake 构建 C++ 矿机二进制
4. 安装 Python 测试依赖：
   ```
   pip3 install aiosqlite fastapi pydantic uvicorn httpx pytest pytest-asyncio
   ```
5. 运行测试（排除 C++ 集成测试）：
   ```
   python3 -m pytest tests/ -v --tb=short --ignore=tests/integration/test_cpp_worker.py
   ```
6. 上传构建产物

### 触发条件

- push 到 `main`
- PR 到 `main`
- 手动触发（workflow_dispatch）

## 开发环境搭建

### Python（服务端 + 测试）

```bash
pip install -r server/requirements.txt
pip install pytest pytest-asyncio httpx paho-mqtt
```

`server/requirements.txt` 包含：

```
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0
aiosqlite>=0.19.0
eth-account>=0.13.0
PyJWT>=2.9.0
```

可选：`pip install argon2-cffi`（启用 Argon2id 相关测试）

### 前端 Dashboard

```bash
cd web && npm install
```

### C++ 矿机（可选）

需要 CUDA 11.8+、CMake 3.29+、vcpkg。详见项目 README。

### 启动 mock 服务端

```bash
python -m server.server --mqtt-port 1883 --api-port 8080
```

CLI 参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--mqtt-port` | 1883 | MQTT broker 端口 |
| `--api-port` | 8080 | REST API 端口 |
| `--db-path` | data/marketplace.db | SQLite 数据库路径 |
| `--no-chain` | false | 禁用内嵌 chain simulator |
| `--block-marker` | XEN11 | 区块检测标记（测试可覆盖） |
| `--jwt-secret` | 自动生成 | JWT 签名密钥（未设置则重启失效） |
