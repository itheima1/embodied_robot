# 机械臂模拟器 WebSocket 控制功能

本文档介绍如何使用 WebSocket 功能来远程控制机械臂模拟器的关节角度。

## 功能概述

机械臂模拟器现在支持通过 WebSocket 接收外部发送的舵机角度数据，实时更新 URDF 模型的显示效果。这使得您可以：

- 从外部程序（Python、C++、其他 Web 应用等）控制机械臂
- 实现实时的机械臂运动控制
- 集成到更大的机器人控制系统中

## 快速开始

### 1. 启动机械臂模拟器

在浏览器中打开 `index.html`，机械臂模拟器会自动尝试连接到 `ws://localhost:8080`。

### 2. 启动 WebSocket 服务器

#### 方法一：使用提供的 Python 示例服务器

```bash
# 安装依赖
pip install websockets

# 运行示例服务器
python websocket_server_example.py
```

示例服务器会自动发送正弦波测试数据，让机械臂执行连续的运动。

#### 方法二：使用测试工具

在浏览器中打开 `websocket_test.html`，这是一个可视化的测试工具，允许您：
- 手动调节每个关节的角度
- 发送预设动作
- 实时查看通信日志

### 3. 观察效果

- 模拟器右上角会显示 WebSocket 连接状态
- 绿色表示已连接，橙色表示断开，红色表示错误
- 机械臂会根据接收到的角度数据实时更新姿态

## 数据格式

### 支持的消息格式

#### 格式1：数组格式（推荐）
```json
{
  "angles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
  "timestamp": 1234567890
}
```

#### 格式2：对象格式（servo键）
```json
{
  "servo1": 0.1,
  "servo2": 0.2,
  "servo3": 0.3,
  "servo4": 0.4,
  "servo5": 0.5,
  "servo6": 0.6
}
```

#### 格式3：对象格式（数字键）
```json
{
  "1": 0.1,
  "2": 0.2,
  "3": 0.3,
  "4": 0.4,
  "5": 0.5,
  "6": 0.6
}
```

### 角度单位

- **弧度制**：推荐使用弧度制（-π 到 π）
- **度数制**：如果角度值的绝对值大于 2π，系统会自动识别为度数并转换为弧度

### 关节映射

| 数组索引 | 关节名称 | 功能描述 |
|---------|---------|----------|
| 0 | joint1 | 基座旋转 |
| 1 | joint2 | 肩部关节 |
| 2 | joint3 | 肘部关节 |
| 3 | joint4 | 腕部关节1 |
| 4 | joint5 | 腕部关节2 |
| 5 | joint6 | 腕部关节3 |

## 编程示例

### Python 客户端示例

```python
import asyncio
import websockets
import json
import math

async def send_angles():
    uri = "ws://localhost:8080"
    
    async with websockets.connect(uri) as websocket:
        # 发送单次角度数据
        angles = [0.5, -0.3, 0.8, 0.0, -0.2, 0.1]
        message = json.dumps({"angles": angles})
        await websocket.send(message)
        
        # 发送连续运动数据
        for t in range(100):
            angles = [math.sin(t * 0.1) * 0.5 for _ in range(6)]
            message = json.dumps({"angles": angles})
            await websocket.send(message)
            await asyncio.sleep(0.1)

# 运行客户端
asyncio.run(send_angles())
```

### JavaScript 客户端示例

```javascript
// 连接到 WebSocket 服务器
const ws = new WebSocket('ws://localhost:8080');

ws.onopen = function() {
    console.log('WebSocket 连接已建立');
    
    // 发送角度数据
    const angles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    const message = JSON.stringify({angles: angles});
    ws.send(message);
};

ws.onmessage = function(event) {
    console.log('收到消息:', event.data);
};
```

### Node.js 服务器示例

```javascript
const WebSocket = require('ws');

const wss = new WebSocket.Server({ port: 8080 });

wss.on('connection', function connection(ws) {
    console.log('新客户端连接');
    
    // 定期发送测试数据
    const interval = setInterval(() => {
        const angles = Array.from({length: 6}, () => Math.random() * 2 - 1);
        ws.send(JSON.stringify({angles: angles}));
    }, 100);
    
    ws.on('close', () => {
        clearInterval(interval);
        console.log('客户端断开连接');
    });
});

console.log('WebSocket 服务器启动在端口 8080');
```

## 高级功能

### 自动重连

模拟器具有自动重连功能：
- 连接断开后会在 5 秒后自动尝试重连
- 重连过程对用户透明
- 状态指示器会实时显示连接状态

### 安全限制

- 所有角度数据都会根据 URDF 文件中定义的关节限制进行约束
- 超出限制的角度会被自动钳制到允许范围内
- 这确保了机械臂不会移动到不安全的位置

### 性能优化

- 建议发送频率不超过 100Hz（每 10ms 一次）
- 过高的发送频率可能导致性能问题
- 系统会自动处理消息队列，避免阻塞

## 故障排除

### 常见问题

1. **连接失败**
   - 检查 WebSocket 服务器是否正在运行
   - 确认端口 8080 没有被其他程序占用
   - 检查防火墙设置

2. **机械臂不动**
   - 确认发送的角度数据格式正确
   - 检查角度值是否在有效范围内
   - 查看浏览器控制台是否有错误信息

3. **连接频繁断开**
   - 检查网络连接稳定性
   - 降低数据发送频率
   - 检查服务器端是否有异常

### 调试技巧

1. **查看连接状态**
   - 观察右上角的状态指示器
   - 打开浏览器开发者工具查看控制台日志

2. **使用测试工具**
   - 使用 `websocket_test.html` 进行手动测试
   - 逐步调试数据格式和通信流程

3. **监控数据流**
   - 在服务器端添加日志输出
   - 使用网络抓包工具分析 WebSocket 通信

## 扩展开发

### 自定义 WebSocket 服务器

您可以根据需要开发自定义的 WebSocket 服务器：

1. **选择合适的编程语言和框架**
   - Python: `websockets`, `tornado`, `flask-socketio`
   - Node.js: `ws`, `socket.io`
   - Java: `Java-WebSocket`, `Spring WebSocket`
   - C++: `websocketpp`, `libwebsockets`

2. **实现必要的功能**
   - 客户端连接管理
   - 消息格式验证
   - 错误处理和日志记录
   - 可选的身份验证和授权

3. **集成到现有系统**
   - 连接到机器人控制系统
   - 集成传感器数据
   - 实现复杂的运动规划算法

### 修改模拟器配置

如果需要修改 WebSocket 连接参数，可以编辑 `index.js` 文件：

```javascript
// 修改 WebSocket 服务器地址
const wsUrl = 'ws://your-server:your-port';

// 修改重连间隔
setTimeout(() => {
    // 重连逻辑
}, 5000); // 5秒改为其他值
```

## 许可证

本项目遵循原项目的许可证条款。WebSocket 功能作为扩展功能提供，可自由使用和修改。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进 WebSocket 功能。在提交之前，请确保：

1. 代码符合项目的编码规范
2. 添加了适当的注释和文档
3. 进行了充分的测试
4. 更新了相关的文档

---

如有任何问题或建议，请通过 GitHub Issues 联系我们。