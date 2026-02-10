# XenBlocks Server REST API

## 认证

支持两种认证方式，`resolve_account()` 优先检查 JWT，其次 API Key。

### 1. 钱包认证（推荐）

基于 EIP-191 签名的认证流程：

```
1. GET  /api/auth/nonce?address=0x...   -> 获取 nonce 和待签名消息
2. 用钱包对返回的 message 进行签名
3. POST /api/auth/verify                -> 提交签名，获取 JWT
4. 后续请求携带 Authorization: Bearer <jwt>
```

- Nonce 有效期：5 分钟
- JWT 有效期：24 小时
- 签名消息格式：`Sign this message to authenticate with XenBlocks.\n\nNonce: {nonce}`

### 2. API Key 认证（兼容旧版）

通过 `POST /api/auth/register` 或 `POST /api/auth/login` 获取 API Key，后续请求携带 `X-API-Key` Header。

Admin Key 通过启动参数 `--admin-key` 配置，拥有 admin 角色的全部权限。

---

## 认证端点 (`/api/auth/*`)

### GET /api/auth/nonce

获取钱包签名用的 nonce。

| 参数 | 位置 | 类型 | 必填 | 说明 |
|------|------|------|------|------|
| address | query | string | 是 | 0x 开头的以太坊地址（42 字符） |

**响应 200：**
```json
{
  "nonce": "a1b2c3...",
  "message": "Sign this message to authenticate with XenBlocks.\n\nNonce: a1b2c3..."
}
```

**错误：** 400 地址格式无效

---

### POST /api/auth/verify

验证钱包签名，返回 JWT。若该地址无账户则自动创建。

**请求 Body：**
```json
{
  "address": "0x...",
  "signature": "0x...",
  "nonce": "a1b2c3..."
}
```

**响应 200：**
```json
{
  "token": "<jwt>",
  "address": "0x...",
  "account_id": "...",
  "role": "provider|consumer"
}
```

**错误：** 401 签名无效或 nonce 过期

---

### POST /api/auth/register

注册新账户（旧版 API Key 流程）。

**请求 Body：**
```json
{
  "account_id": "string",
  "role": "provider|consumer",
  "eth_address": "",
  "balance": 0.0
}
```

**响应 200：**
```json
{
  "account_id": "...",
  "role": "...",
  "eth_address": "...",
  "balance": 0.0,
  "api_key": "generated-key"
}
```

**错误：** 400 角色无效或账户已存在

---

### POST /api/auth/login

登录已有账户，获取 API Key（无 Key 则自动生成）。

**请求 Body：**
```json
{
  "account_id": "string"
}
```

**响应 200：**
```json
{
  "account_id": "...",
  "role": "...",
  "api_key": "..."
}
```

**错误：** 404 账户不存在

---

### GET /api/auth/me

获取当前认证账户信息。**需要认证。**

**响应 200：**
```json
{
  "account_id": "...",
  "role": "...",
  "eth_address": "...",
  "balance": 0.0
}
```

**错误：** 401 未认证

---

## 账户端点 (`/api/accounts/*`)

### GET /api/accounts/{account_id}/balance

查询账户余额。认证用户只能查看自己的余额（admin 除外）。

| 参数 | 位置 | 类型 | 必填 | 说明 |
|------|------|------|------|------|
| account_id | path | string | 是 | 账户 ID |

**响应 200：**
```json
{
  "account_id": "...",
  "role": "...",
  "balance": 0.0,
  "eth_address": "..."
}
```

**错误：** 403 无权限 / 404 账户不存在

---

### POST /api/accounts/{account_id}/deposit

向账户充值。认证用户只能向自己充值（admin 除外）。

| 参数 | 位置 | 类型 | 必填 | 说明 |
|------|------|------|------|------|
| account_id | path | string | 是 | 账户 ID |

**请求 Body：**
```json
{
  "amount": 10.0
}
```

**响应 200：**
```json
{
  "account_id": "...",
  "balance": 10.0
}
```

**错误：** 400 金额无效 / 403 无权限 / 404 账户不存在

---

## 市场端点 (`/api/marketplace/*`, `/api/workers/*`)

### GET /api/workers

列出所有可用 worker，附带信誉评分。无需认证。

**响应 200：** Worker 对象数组，每个包含 `reputation` 字段。

---

### GET /api/marketplace

浏览算力市场，支持筛选和排序。无需认证。

| 参数 | 位置 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|------|--------|------|
| sort_by | query | string | 否 | "price" | 排序字段 |
| gpu_type | query | string | 否 | - | GPU 类型筛选 |
| min_hashrate | query | float | 否 | - | 最低算力 |
| max_price | query | float | 否 | - | 最高价格 |
| min_gpus | query | int | 否 | - | 最少 GPU 数量 |
| available_only | query | bool | 否 | true | 仅显示可用 |

**响应 200：** 市场列表数据。

---

### GET /api/marketplace/estimate

估算租赁成本。无需认证。

| 参数 | 位置 | 类型 | 必填 | 说明 |
|------|------|------|------|------|
| duration_sec | query | int | 是 | 租赁时长（秒） |
| worker_id | query | string | 否 | 指定 worker |
| min_hashrate | query | float | 否 | 最低算力要求 |

**响应 200：** 成本估算结果。

**错误：** 404 无匹配 worker

---

### GET /api/workers/{worker_id}/pricing

查询 worker 定价信息。无需认证。

**响应 200：** 定价对象。

**错误：** 404 Worker 不存在

---

### GET /api/workers/{worker_id}/pricing/suggest

获取 worker 建议定价。无需认证。

**响应 200：** 建议定价对象。

**错误：** 404 Worker 不存在

---

### GET /api/workers/{worker_id}/reputation

查询 worker 信誉评分。无需认证。

**响应 200：** 信誉评分对象。

**错误：** 404 Worker 不存在

---

### PUT /api/workers/{worker_id}/pricing

设置 worker 定价。需要 provider 或 admin 认证（通过 `X-API-Key`）。Provider 只能设置自己的 worker。

**请求 Body：**
```json
{
  "price_per_min": 0.5,
  "min_duration_sec": 60,
  "max_duration_sec": 86400
}
```

**响应 200：**
```json
{
  "worker_id": "...",
  "price_per_min": 0.5,
  "min_duration_sec": 60,
  "max_duration_sec": 86400
}
```

**错误：** 400 参数无效 / 403 无权限 / 404 Worker 不存在

---

## Provider 端点 (`/api/provider/*`)

所有 Provider 端点通过 `provider_id` query 参数或 JWT 中的钱包地址标识 provider。

### GET /api/provider/dashboard

Provider 仪表盘概览。

| 参数 | 位置 | 类型 | 必填 | 说明 |
|------|------|------|------|------|
| provider_id | query | string | 否 | Provider ID / 钱包地址 / worker_id |
| Authorization | header | string | 否 | Bearer JWT（与 provider_id 二选一） |

**响应 200：**
```json
{
  "provider_id": "...",
  "worker_count": 3,
  "total_earned": 1.5,
  "active_leases": 1,
  "total_blocks_mined": 42,
  "avg_hashrate": 120.5
}
```

**错误：** 400 缺少 provider_id 且无 JWT

---

### GET /api/provider/earnings

查询 provider 结算收入。参数同 dashboard。

**响应 200：**
```json
{
  "provider_id": "...",
  "earnings": [{ "...settlement objects..." }]
}
```

---

### GET /api/provider/workers

列出 provider 的 worker 列表及状态。参数同 dashboard。

**响应 200：**
```json
{
  "provider_id": "...",
  "workers": [
    {
      "worker_id": "...",
      "state": "idle",
      "online": true,
      "hashrate": 100.0,
      "gpu_count": 2,
      "active_gpus": 2,
      "price_per_min": 0.5,
      "self_blocks_found": 10,
      "total_online_sec": 36000
    }
  ]
}
```

---

### PUT /api/provider/workers/{worker_id}/pricing

设置 provider worker 的定价。需要 provider 或 admin 认证。Provider 只能设置自己的 worker。

请求/响应格式同 `PUT /api/workers/{worker_id}/pricing`。

---

## 租赁端点 (`/api/rent`, `/api/leases/*`, `/api/blocks/*`)

### POST /api/rent

租赁算力。需要 consumer 或 admin 角色（通过 `X-API-Key`）。

**请求 Body：**
```json
{
  "consumer_id": "string",
  "consumer_address": "0x...",
  "duration_sec": 3600,
  "worker_id": null
}
```

**响应 200：**
```json
{
  "lease_id": "...",
  "worker_id": "...",
  "prefix": "...",
  "duration_sec": 3600,
  "consumer_id": "...",
  "consumer_address": "0x...",
  "created_at": 1700000000
}
```

**错误：** 403 非 consumer 角色 / 404 无可用 worker

---

### POST /api/stop

停止租约并结算。需要 consumer 或 admin 角色。Consumer 只能停止自己的租约。

**请求 Body：**
```json
{
  "lease_id": "string"
}
```

**响应 200：**
```json
{
  "lease_id": "...",
  "state": "completed",
  "blocks_found": 5,
  "settlement": { "...settlement object..." }
}
```

**错误：** 403 无权限 / 404 租约不存在或非活跃

---

### POST /api/rental/start

`POST /api/rent` 的别名，参数和响应相同。

### POST /api/rental/stop

`POST /api/stop` 的别名，参数和响应相同。

---

### GET /api/leases

列出租约，支持分页和状态过滤。无需认证。

| 参数 | 位置 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|------|--------|------|
| state | query | string | 否 | - | 状态过滤 (active/completed/cancelled) |
| limit | query | int | 否 | 50 | 每页数量 |
| offset | query | int | 否 | 0 | 偏移量 |

**响应 200：**
```json
{
  "items": [...],
  "total": 100,
  "limit": 50,
  "offset": 0
}
```

---

### GET /api/leases/{lease_id}

查询租约详情，包含区块和结算信息。无需认证。

**响应 200：**
```json
{
  "lease_id": "...",
  "worker_id": "...",
  "consumer_id": "...",
  "consumer_address": "...",
  "prefix": "...",
  "duration_sec": 3600,
  "state": "active",
  "created_at": 0,
  "ended_at": 0,
  "blocks_found": 0,
  "avg_hashrate": 0.0,
  "elapsed_sec": 0,
  "blocks": [],
  "settlement": {}
}
```

**错误：** 404 租约不存在

---

### GET /api/blocks

列出区块，支持分页和按租约过滤。无需认证。

| 参数 | 位置 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|------|--------|------|
| lease_id | query | string | 否 | - | 按租约过滤 |
| limit | query | int | 否 | 50 | 每页数量 |
| offset | query | int | 否 | 0 | 偏移量 |

**响应 200：** 分页区块列表 `{ items, total, limit, offset }`，指定 `lease_id` 时直接返回区块数组。

---

### GET /api/blocks/self-mined

列出自挖区块。无需认证。

| 参数 | 位置 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|------|--------|------|
| worker_id | query | string | 否 | - | 按 worker 过滤 |
| limit | query | int | 否 | 50 | 每页数量 |
| offset | query | int | 否 | 0 | 偏移量 |

**响应 200：** `{ items, total, limit, offset }`

---

## 监控端点 (`/api/monitoring/*`)

所有监控端点无需认证。

### GET /api/monitoring/fleet

获取矿机集群概览。

---

### GET /api/monitoring/stats

获取聚合统计数据。

---

### GET /api/monitoring/hashrate-history

获取算力历史。

| 参数 | 位置 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|------|--------|------|
| worker_id | query | string | 否 | - | 指定 worker（不指定则全局） |
| hours | query | float | 否 | 1.0 | 时间范围（0.0167~24.0 小时） |

---

### GET /api/monitoring/blocks/recent

获取最近挖出的区块。

| 参数 | 位置 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|------|--------|------|
| limit | query | int | 否 | 20 | 返回数量（1~200） |

---

## 全局概览端点 (`/api/overview/*`)

所有概览端点无需认证。

### GET /api/overview/stats

平台全局统计。

**响应 200：**
```json
{
  "total_users": 10,
  "total_providers": 5,
  "total_consumers": 5,
  "total_workers": 8,
  "online_workers": 6,
  "total_hashrate": 800.0,
  "total_blocks": 1000,
  "blocks_24h": 50,
  "total_leases": 200,
  "active_leases": 3,
  "total_settled": 150,
  "platform_revenue": 12.5
}
```

---

### GET /api/overview/activity

最近平台活动（区块、租约事件），最多 50 条，按时间倒序。

**响应 200：**
```json
[
  {
    "type": "block|lease_started|lease_completed",
    "timestamp": 1700000000,
    "details": { "..." }
  }
]
```

---

### GET /api/overview/network

网络状态信息。

**响应 200：**
```json
{
  "difficulty": 12345,
  "total_workers": 8,
  "total_blocks": 1000,
  "chain_blocks": 500
}
```

---

## 管理端点

### GET /

服务基本信息。无需认证。

**响应 200：**
```json
{
  "service": "XenMiner Mock Platform",
  "mqtt_port": 1883,
  "api_port": 8000,
  "connected_workers": 5,
  "uptime": "running"
}
```

---

### GET /api/status

服务状态。无需认证。

**响应 200：**
```json
{
  "mqtt_clients": ["worker-1", "worker-2"],
  "workers": 5,
  "active_leases": 2,
  "total_blocks": 100,
  "self_mined_blocks": 10,
  "total_settlements": 50
}
```

---

### GET /api/accounts

列出所有账户。需要 admin 角色（通过 `X-API-Key`）。

**响应 200：** `{ "account_id": { account_id, role, eth_address, balance }, ... }`

**错误：** 403 非 admin

---

### GET /api/settlements

列出结算记录。需要 admin 角色。

| 参数 | 位置 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|------|--------|------|
| limit | query | int | 否 | 50 | 每页数量 |
| offset | query | int | 否 | 0 | 偏移量 |

**响应 200：** `{ items, total, limit, offset }`

**错误：** 403 非 admin

---

### POST /api/workers/{worker_id}/control

向指定 worker 发送控制指令。无需认证。

**请求 Body：**
```json
{
  "action": "set_config",
  "config": {}
}
```

**响应 200：**
```json
{
  "status": "sent",
  "worker_id": "...",
  "action": "set_config"
}
```

---

### POST /api/control/broadcast

向所有可用 worker 广播控制指令。无需认证。

**请求 Body：** 同上。

**响应 200：**
```json
{
  "status": "sent",
  "workers": ["worker-1", "worker-2"],
  "action": "set_config"
}
```

---

## WebSocket

### WS /ws/dashboard

实时仪表盘数据推送。连接后由 `ws_manager` 处理。服务不可用时返回 1013 关闭码。
