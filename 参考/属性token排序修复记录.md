# 属性Token排序修复记录

## 问题描述
用户发现属性token的排序可能存在问题，需要与C++和Python版本进行对比验证。

## 问题分析
通过对比分析发现：

### C++版本排序（正确）
在 `convert_standard_properties_to_tokens` 函数中：
1. tts_special_token_offset
2. age
3. gender
4. emotion
5. pitch
6. speed

### Rust版本排序
原本已经与C++版本保持一致，但发现了一个关键差异：

**儿童女性音高分类差异：**
- C++版本：儿童女性只有三个音高级别（low_pitch, medium_pitch, high_pitch）
- Rust版本（修复前）：儿童女性有四个音高级别（包括very_high_pitch）

## 修复内容

### 1. 修复儿童女性音高分类
在 `src/properties_util.rs` 中的 `classify_pitch` 函数：

```rust
// 修复前
"child" => {
    if pitch < 250.0 {
        "low_pitch".to_string()
    } else if pitch < 290.0 {
        "medium_pitch".to_string()
    } else if pitch < 330.0 {
        "high_pitch".to_string()
    } else {
        "very_high_pitch".to_string()
    }
}

// 修复后
"child" => {
    if pitch < 250.0 {
        "low_pitch".to_string()
    } else if pitch < 290.0 {
        "medium_pitch".to_string()
    } else {
        "high_pitch".to_string()
    }
}
```

### 2. 更新相关测试用例
修改了 `test_child_female_very_high_pitch` 测试函数，改名为 `test_child_female_pitch_classification`，验证儿童女性只有三个音高级别。

## 验证结果

### 测试通过情况
- ✅ 所有原有测试用例通过
- ✅ 新增兼容性测试用例通过
- ✅ 属性token排序与C++版本完全一致
- ✅ 儿童女性音高分类与C++版本完全一致

### 代码质量检查
- ✅ `cargo fmt` 格式化通过
- ✅ `cargo clippy` 检查通过（有少量警告但不影响功能）

## 修复时间
2024年12月（具体日期根据实际修复时间）

## 影响范围
此修复确保了Rust版本与C++版本在属性token生成方面的完全兼容性，特别是：
1. 属性token的排列顺序完全一致
2. 儿童女性音高分类逻辑完全一致
3. 所有其他属性分类逻辑保持一致

## 重要性
这个修复对于确保不同语言版本之间的模型兼容性至关重要，避免了因属性token差异导致的推理结果不一致问题。