# 更新日志

## [0.2.3] - 2025-09-17

### 修复
- 修复了Web UI中的JavaScript语法错误，解决了"Uncaught SyntaxError: Unexpected token ':'"问题
- 移除了不再需要的`handle_tts_with_file_upload`函数，简化了TTS处理逻辑
- 移除了未使用的`TtsRequest`结构体

### 改进
- 简化了TTS请求处理流程，现在只支持通过voice_id使用预存的音色特征
- 清理了冗余代码，提高了代码可维护性

## [0.2.2] - 2025-09-17

### 新增
- 添加了Web界面截图到README文档
- 创建了中英文README文档并相互链接

### 修复
- 修复了GitHub Actions构建脚本中raf目录路径问题
- 修复了Web UI删除音色功能的HTTP方法不匹配问题
- 统一了不同语言实现中的属性拼接顺序（age, gender, emotion, pitch, speed）
- 调整了Web UI参数部分：随机种子放在第一个，语速使用下拉框对应5个选项，默认是medium中速

### 改进
- 更新了API处理speed参数的逻辑，支持字符串类型（"very_slow", "slow", "medium", "fast", "very_fast"）

## [0.2.1] - 2025-09-17

### 新增
- 简化了README文档，只保留核心信息（项目简介、编译运行说明、预编译版本获取流程）
- 创建了英文README文档并相互链接

### 修复
- 修复了GitHub Actions构建脚本中raf目录没有正确添加到assets目录下的问题

## [0.2.0] - 2025-09-16

### 新增
- 实现了音色克隆功能，支持提取和使用自定义音色特征
- 添加了Web UI界面，提供图形化操作体验
- 实现了多语言支持（中文/英文）
- 添加了高级TTS参数控制（年龄、性别、情感、音调等）

### 改进
- 优化了TTS生成性能
- 改进了错误处理和日志记录
- 增强了API文档和使用说明