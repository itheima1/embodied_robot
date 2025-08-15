import {
  WebGLRenderer,
  PerspectiveCamera,
  Scene,
  Mesh,
  PlaneGeometry,
  ShadowMaterial,
  DirectionalLight,
  PCFSoftShadowMap,
  // sRGBEncoding,
  Color,
  AmbientLight,
  Box3,
  LoadingManager,
  MathUtils,
  MeshPhysicalMaterial,
  DoubleSide,
  ACESFilmicToneMapping,
  CanvasTexture,
  Float32BufferAttribute,
  RepeatWrapping,
} from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import URDFLoader from 'urdf-loader';
// 导入控制工具函数
import { setupKeyboardControls, setupControlPanel } from './robotControls.js';

// 声明为全局变量
let scene, camera, renderer, controls;
// 将robot设为全局变量，便于其他模块访问
window.robot = null;
let keyboardUpdate;

// WebSocket相关变量
let websocket = null;
let isWebSocketConnected = false;

init();
render();

function init() {

  scene = new Scene();
  scene.background = new Color(0x263238);

  camera = new PerspectiveCamera();
  camera.position.set(5, 5, 5);
  camera.lookAt(0, 0, 0);

  renderer = new WebGLRenderer({ antialias: true });
  // renderer.outputEncoding = sRGBEncoding;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = PCFSoftShadowMap;
  renderer.physicallyCorrectLights = true;
  renderer.toneMapping = ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.5;
  document.body.appendChild(renderer.domElement);

  const directionalLight = new DirectionalLight(0xffffff, 1.0);
  directionalLight.castShadow = true;
  directionalLight.shadow.mapSize.setScalar(1024);
  directionalLight.position.set(5, 30, 5);
  scene.add(directionalLight);

  // Add second directional light for better reflections
  const directionalLight2 = new DirectionalLight(0xffffff, 0.8);
  directionalLight2.position.set(-2, 10, -5);
  scene.add(directionalLight2);

  const ambientLight = new AmbientLight(0xffffff, 0.3);
  scene.add(ambientLight);

  // Create reflective floor (MuJoCo style)
  const groundMaterial = new MeshPhysicalMaterial({
    color: 0x808080,
    metalness: 0.7,
    roughness: 0.3,
    reflectivity: 0.1,
    clearcoat: 0.3,
    side: DoubleSide,
    transparent: true,     // 启用透明度
    opacity: 0.7,          // 设置透明度为0.7（可以根据需要调整，1.0为完全不透明）
  });
  
  // 创建格子纹理的地面
  const gridSize = 60;
  const divisions = 60;
  
  // 创建网格地面
  const ground = new Mesh(new PlaneGeometry(gridSize, gridSize, divisions, divisions), groundMaterial);
  
  // 添加格子纹理
  const geometry = ground.geometry;
  const positionAttribute = geometry.getAttribute('position');
  
  // 创建格子纹理的UV坐标
  const uvs = [];
  const gridScale = 0.01; // 控制格子的密度
  
  for (let i = 0; i < positionAttribute.count; i++) {
    const x = positionAttribute.getX(i);
    const y = positionAttribute.getY(i);
    
    uvs.push(x * gridScale, y * gridScale);
  }
  
  geometry.setAttribute('uv', new Float32BufferAttribute(uvs, 2));
  
  // 更新材质，添加格子纹理
  groundMaterial.map = createGridTexture();
  groundMaterial.roughnessMap = createGridTexture();
  
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  scene.add(ground);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.minDistance = 4;
  controls.target.y = 1;
  controls.update();

  // 根据URL hash或默认加载模型
  function loadModelFromHash() {
    // 获取URL hash（去掉#号）
    let modelToLoad = 'genkiarm';
    
    // 加载模型
    const manager = new LoadingManager();
    const loader = new URDFLoader(manager);

    loader.load(`/URDF/${modelToLoad}.urdf`, result => {
      window.robot = result;
    });

    // 等待模型加载完成
    manager.onLoad = () => {
      window.robot.rotation.x = - Math.PI / 2;
      window.robot.rotation.z = - Math.PI;
      window.robot.traverse(c => {
        c.castShadow = true;
      });
      console.log(window.robot.joints);
      // 记录关节限制信息到控制台，便于调试
      logJointLimits(window.robot);
      
      window.robot.updateMatrixWorld(true);

      const bb = new Box3();
      bb.setFromObject(window.robot);

      window.robot.scale.set(15, 15, 15);
      window.robot.position.y -= bb.min.y;
      scene.add(window.robot);

      // Initialize keyboard controls
      keyboardUpdate = setupKeyboardControls(window.robot);
      
      // Initialize WebSocket connection after robot is loaded
      initWebSocket();
    };
  }

  // 初始加载模型
  loadModelFromHash();

  onResize();
  window.addEventListener('resize', onResize);

  // Setup UI for control panel
  setupControlPanel();
}

/**
 * 输出关节限制信息到控制台
 * @param {Object} robot - 机器人对象
 */
function logJointLimits(robot) {
  if (!robot || !robot.joints) return;
  
  console.log("Robot joint limits:");
  Object.entries(robot.joints).forEach(([name, joint]) => {
    console.log(`Joint: ${name}`);
    console.log(`  Type: ${joint.jointType}`);
    
    if (joint.jointType !== 'fixed' && joint.jointType !== 'continuous') {
      console.log(`  Limits: ${joint.limit.lower.toFixed(4)} to ${joint.limit.upper.toFixed(4)} rad`);
      console.log(`  Current value: ${Array.isArray(joint.jointValue) ? joint.jointValue.join(', ') : joint.jointValue}`);
    } else if (joint.jointType === 'continuous') {
      console.log(`  No limits (continuous joint)`);
    } else {
      console.log(`  No limits (fixed joint)`);
    }
  });
}

function onResize() {
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);

  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
}

function render() {
  requestAnimationFrame(render);
  
  // Update joint positions based on keyboard input
  if (keyboardUpdate) {
    keyboardUpdate();
  }
  
  renderer.render(scene, camera);
}

// 添加创建格子纹理的函数
function createGridTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  
  const context = canvas.getContext('2d');
  
  // 填充底色
  context.fillStyle = '#808080';
  context.fillRect(0, 0, canvas.width, canvas.height);
  
  // 绘制格子线
  context.lineWidth = 1;
  context.strokeStyle = '#606060';
  
  const cellSize = 32; // 每个格子的大小
  
  for (let i = 0; i <= canvas.width; i += cellSize) {
    context.beginPath();
    context.moveTo(i, 0);
    context.lineTo(i, canvas.height);
    context.stroke();
  }
  
  for (let i = 0; i <= canvas.height; i += cellSize) {
    context.beginPath();
    context.moveTo(0, i);
    context.lineTo(canvas.width, i);
    context.stroke();
  }
  
  // 修复: 使用已导入的 CanvasTexture，而不是 THREE.CanvasTexture
  const texture = new CanvasTexture(canvas);
  // 修复: 使用已导入的 RepeatWrapping，而不是 THREE.RepeatWrapping
  texture.wrapS = RepeatWrapping;
  texture.wrapT = RepeatWrapping;
  texture.repeat.set(10, 10);
  
  return texture;
}

/**
 * 初始化WebSocket连接
 */
function initWebSocket() {
  // 默认连接到本地WebSocket服务器
  const wsUrl = 'ws://localhost:8080';
  
  try {
    websocket = new WebSocket(wsUrl);
    
    websocket.onopen = function(event) {
      console.log('WebSocket连接已建立:', wsUrl);
      isWebSocketConnected = true;
      updateWebSocketStatus('connected');
    };
    
    websocket.onmessage = function(event) {
       console.log('收到WebSocket原始数据:', event.data);
       try {
         const data = JSON.parse(event.data);
         console.log('解析后的数据:', data);
         console.log('数据类型:', typeof data, '是否为数组:', Array.isArray(data));
         handleServoAngleData(data);
       } catch (error) {
         console.error('解析WebSocket消息失败:', error);
         console.error('原始数据:', event.data);
       }
     };
    
    websocket.onclose = function(event) {
      console.log('WebSocket连接已关闭');
      isWebSocketConnected = false;
      updateWebSocketStatus('disconnected');
      
      // 5秒后尝试重连
      setTimeout(() => {
        if (!isWebSocketConnected) {
          console.log('尝试重新连接WebSocket...');
          initWebSocket();
        }
      }, 5000);
    };
    
    websocket.onerror = function(error) {
      console.error('WebSocket错误:', error);
      updateWebSocketStatus('error');
    };
    
  } catch (error) {
    console.error('创建WebSocket连接失败:', error);
    updateWebSocketStatus('error');
  }
}

/**
 * 处理接收到的舵机角度数据
 * @param {Object} data - 包含舵机角度的数据对象
 */
function handleServoAngleData(data) {
  console.log('开始处理伺服角度数据:', data);
  
  if (!window.robot || !window.robot.joints) {
    console.warn('机器人模型未加载，无法更新关节角度');
    return;
  }
  
  // 期望的数据格式: { angles: [angle1, angle2, angle3, angle4, angle5, angle6] }
  // 或者: { servo1: angle1, servo2: angle2, ... }
  
  let angles = [];
  
  if (data.angles && Array.isArray(data.angles)) {
    // 数组格式
    angles = data.angles;
    console.log('识别为angles数组格式，角度值:', angles);
  } else if (Array.isArray(data)) {
    // 直接数组格式
    angles = data;
    console.log('识别为直接数组格式，角度值:', angles);
  } else if (typeof data === 'object') {
    // 对象格式，提取servo1-servo6或1-6的值
    console.log('识别为对象格式，开始解析各个关节角度');
    for (let i = 1; i <= 6; i++) {
      const servoKey = `servo${i}`;
      const numKey = i.toString();
      
      if (data[servoKey] !== undefined) {
        angles[i-1] = data[servoKey];
        console.log(`解析关节${i} (${servoKey}): ${data[servoKey]}`);
      } else if (data[numKey] !== undefined) {
        angles[i-1] = data[numKey];
        console.log(`解析关节${i} (${numKey}): ${data[numKey]}`);
      } else {
        angles[i-1] = 0; // 默认值
        console.log(`关节${i}未找到数据，使用默认值0`);
      }
    }
  } else {
    console.error('不支持的数据格式:', typeof data, data);
    return;
  }
  
  console.log('准备更新机器人关节，最终角度数组:', angles, '长度:', angles.length);
  // 更新机器人关节角度
  updateRobotJoints(angles);
}

/**
 * 更新机器人关节角度
 * @param {Array} angles - 6个关节的角度数组（弧度制）
 */
function updateRobotJoints(angles) {
  console.log('开始更新机器人关节，输入角度:', angles);
  
  if (!window.robot || !window.robot.joints || angles.length < 6) {
    console.warn('机器人模型未加载或角度数组长度不足');
    return;
  }
  
  console.log('可用关节列表:', Object.keys(window.robot.joints));
  
  // 关节名称映射（根据URDF文件中的关节名称）
  const jointNames = [
    'Rotation',   // 基座旋转
    'Rotation2',  // 肩部
    'Rotation3',  // 肘部
    'Rotation4',  // 腕部1
    'Rotation5',  // 腕部2
    'Rotation6'   // 腕部3
  ];
  
  // 更新每个关节的角度
  for (let i = 0; i < Math.min(jointNames.length, angles.length); i++) {
    const jointName = jointNames[i];
    const joint = window.robot.joints[jointName];
    
    console.log(`处理关节 ${i+1}: ${jointName}`);
    console.log(`关节对象:`, joint ? '找到' : '未找到');
    
    if (joint) {
      // 直接使用弧度制角度
      let angleRad = parseFloat(angles[i]) || 0;
      console.log(`原始角度值: ${angles[i]}, 解析后: ${angleRad}`);
      
      // 检查关节限制
      if (joint.limit) {
        const originalAngle = angleRad;
        angleRad = MathUtils.clamp(angleRad, joint.limit.lower, joint.limit.upper);
        console.log(`关节限制: [${joint.limit.lower.toFixed(3)}, ${joint.limit.upper.toFixed(3)}]`);
        if (originalAngle !== angleRad) {
          console.log(`角度被限制: ${originalAngle.toFixed(3)} -> ${angleRad.toFixed(3)}`);
        }
      }
      
      // 设置关节角度
      joint.setJointValue(angleRad);
      console.log(`✓ 成功设置关节 ${jointName} 角度: ${angleRad.toFixed(3)} 弧度`);
    } else {
      console.warn(`✗ 未找到关节: ${jointName}`);
    }
  }
  
  // 更新机器人的变换矩阵
  if (window.robot.updateMatrixWorld) {
    window.robot.updateMatrixWorld(true);
    console.log('✓ 机器人变换矩阵更新完成');
  }
}

/**
 * 更新WebSocket连接状态显示
 * @param {string} status - 连接状态: 'connected', 'disconnected', 'error'
 */
function updateWebSocketStatus(status) {
  // 创建或更新状态显示元素
  let statusElement = document.getElementById('websocket-status');
  
  if (!statusElement) {
    statusElement = document.createElement('div');
    statusElement.id = 'websocket-status';
    statusElement.style.cssText = `
      position: fixed;
      top: 10px;
      right: 10px;
      padding: 8px 12px;
      border-radius: 4px;
      color: white;
      font-family: Arial, sans-serif;
      font-size: 12px;
      z-index: 1000;
      transition: all 0.3s ease;
    `;
    document.body.appendChild(statusElement);
  }
  
  // 根据状态设置样式和文本
  switch (status) {
    case 'connected':
      statusElement.style.backgroundColor = '#4CAF50';
      statusElement.textContent = 'WebSocket: 已连接';
      break;
    case 'disconnected':
      statusElement.style.backgroundColor = '#FF9800';
      statusElement.textContent = 'WebSocket: 已断开';
      break;
    case 'error':
      statusElement.style.backgroundColor = '#F44336';
      statusElement.textContent = 'WebSocket: 连接错误';
      break;
    default:
      statusElement.style.backgroundColor = '#9E9E9E';
      statusElement.textContent = 'WebSocket: 未知状态';
  }
}

// 导出函数供外部使用
window.updateRobotJoints = updateRobotJoints;
window.initWebSocket = initWebSocket;
