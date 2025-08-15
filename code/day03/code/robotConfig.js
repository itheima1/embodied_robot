// 机器人配置文件 - 定义不同机器人类型的配置

// 基础机器人配置
const robotConfigs = {
  // so_arm100 配置
  so_arm100: {
    name: 'SO_ARM100',
    type: 'arm',
    servos: {
      arm: 6, // 机械臂有6个舵机
    },
    // 视角和缩放配置
    viewConfig: {
      camera: {
        position: [0, 5, 10],  // 摄像机位置 [x, y, z]
        rotation: [0, 0, 0],   // 摄像机旋转角度 [x, y, z]
        fov: 45,               // 视场角(度)
        zoom: 1.0              // 缩放比例
      },
      defaultDistance: 15,     // 默认观察距离
      minDistance: 5,          // 最小缩放距离
      maxDistance: 30          // 最大缩放距离
    },
    // 控制映射配置
    controlMapping: {
      arm: {
        type: 'default',  // 默认的机械臂控制方式
        keyMapping: {
    '1': { jointIndex: 0, direction: -1 },
    'q': { jointIndex: 0, direction: 1 },
    '2': { jointIndex: 1, direction: -1 },
    'w': { jointIndex: 1, direction: 1 },
    '3': { jointIndex: 2, direction: 1 },
    'e': { jointIndex: 2, direction: -1 },
    '4': { jointIndex: 3, direction: 1 },
    'r': { jointIndex: 3, direction: -1 },
    '5': { jointIndex: 4, direction: 1 },
    't': { jointIndex: 4, direction: -1 },
    '6': { jointIndex: 5, direction: 1 },
    'y': { jointIndex: 5, direction: -1 },
        }
      }
    }
  }
};

export default robotConfigs; 