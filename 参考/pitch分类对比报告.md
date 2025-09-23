# Pitch分类对比报告

## 概述
对比C++版本和Rust版本的classify_pitch函数实现，检查所有年龄段和性别组合的音高分类阈值。

## 详细对比结果

### 1. 女性分类对比

#### 女性儿童 (child)
- **C++版本**: 
  - < 250.0: low_pitch
  - < 290.0: medium_pitch
  - >= 290.0: high_pitch
- **Rust版本**: 
  - < 250.0: low_pitch
  - < 290.0: medium_pitch
  - >= 290.0: high_pitch
- **结果**: ✅ **完全一致**

#### 女性青少年 (teenager)
- **C++版本**: 
  - < 208.0: low_pitch
  - < 238.0: medium_pitch
  - < 270.0: high_pitch
  - >= 270.0: very_high_pitch
- **Rust版本**: 
  - < 208.0: low_pitch
  - < 238.0: medium_pitch
  - < 270.0: high_pitch
  - >= 270.0: very_high_pitch
- **结果**: ✅ **完全一致**

#### 女性青年成人 (youth-adult)
- **C++版本**: 
  - < 191.0: low_pitch
  - < 211.0: medium_pitch
  - < 232.0: high_pitch
  - >= 232.0: very_high_pitch
- **Rust版本**: 
  - < 191.0: low_pitch
  - < 211.0: medium_pitch
  - < 232.0: high_pitch
  - >= 232.0: very_high_pitch
- **结果**: ✅ **完全一致**

#### 女性中年 (middle-aged)
- **C++版本**: 
  - < 176.0: low_pitch
  - < 195.0: medium_pitch
  - < 215.0: high_pitch
  - >= 215.0: very_high_pitch
- **Rust版本**: 
  - < 176.0: low_pitch
  - < 195.0: medium_pitch
  - < 215.0: high_pitch
  - >= 215.0: very_high_pitch
- **结果**: ✅ **完全一致**

#### 女性老年 (elderly)
- **C++版本**: 
  - < 170.0: low_pitch
  - < 190.0: medium_pitch
  - < 213.0: high_pitch
  - >= 213.0: very_high_pitch
- **Rust版本**: 
  - < 170.0: low_pitch
  - < 190.0: medium_pitch
  - < 213.0: high_pitch
  - >= 213.0: very_high_pitch
- **结果**: ✅ **完全一致**

#### 女性默认分类
- **C++版本**: 
  - < 187.0: low_pitch
  - < 209.0: medium_pitch
  - < 232.0: high_pitch
  - >= 232.0: very_high_pitch
- **Rust版本**: 
  - < 187.0: low_pitch
  - < 209.0: medium_pitch
  - < 232.0: high_pitch
  - >= 232.0: very_high_pitch
- **结果**: ✅ **完全一致**

### 2. 男性分类对比

#### 男性儿童 (child)
- **C++版本**: ❌ **未实现** (没有男性儿童的分类)
- **Rust版本**: ❌ **未实现** (没有男性儿童的分类)
- **结果**: ✅ **一致** (都没有实现)

#### 男性青少年 (teenager)
- **C++版本**: 
  - < 121.0: low_pitch
  - < 143.0: medium_pitch
  - < 166.0: high_pitch
  - >= 166.0: very_high_pitch
- **Rust版本**: 
  - < 121.0: low_pitch
  - < 143.0: medium_pitch
  - < 166.0: high_pitch
  - >= 166.0: very_high_pitch
- **结果**: ✅ **完全一致**

#### 男性青年成人 (youth-adult)
- **C++版本**: 
  - < 115.0: low_pitch
  - < 131.0: medium_pitch
  - < 153.0: high_pitch
  - >= 153.0: very_high_pitch
- **Rust版本**: 
  - < 115.0: low_pitch
  - < 131.0: medium_pitch
  - < 153.0: high_pitch
  - >= 153.0: very_high_pitch
- **结果**: ✅ **完全一致**

#### 男性中年 (middle-aged)
- **C++版本**: 
  - < 110.0: low_pitch
  - < 125.0: medium_pitch
  - < 147.0: high_pitch
  - >= 147.0: very_high_pitch
- **Rust版本**: 
  - < 110.0: low_pitch
  - < 125.0: medium_pitch
  - < 147.0: high_pitch
  - >= 147.0: very_high_pitch
- **结果**: ✅ **完全一致**

#### 男性老年 (elderly)
- **C++版本**: 
  - < 115.0: low_pitch
  - < 128.0: medium_pitch
  - < 142.0: high_pitch
  - >= 142.0: very_high_pitch
- **Rust版本**: 
  - < 115.0: low_pitch
  - < 128.0: medium_pitch
  - < 142.0: high_pitch
  - >= 142.0: very_high_pitch
- **结果**: ✅ **完全一致**

#### 男性默认分类
- **C++版本**: 
  - < 114.0: low_pitch
  - < 130.0: medium_pitch
  - < 151.0: high_pitch
  - >= 151.0: very_high_pitch
- **Rust版本**: 
  - < 130.0: low_pitch
  - < 180.0: medium_pitch
  - < 220.0: high_pitch
  - >= 220.0: very_high_pitch
- **结果**: ❌ **存在差异**

### 3. 未知性别分类对比

- **C++版本**: 
  - < 130.0: low_pitch
  - < 180.0: medium_pitch
  - < 220.0: high_pitch
  - >= 220.0: very_high_pitch
- **Rust版本**: 
  - < 130.0: low_pitch
  - < 180.0: medium_pitch
  - < 220.0: high_pitch
  - >= 220.0: very_high_pitch
- **结果**: ✅ **完全一致**

## 发现的差异

### 关键差异：男性默认分类

**问题位置**: Rust版本的男性默认分类 (`_ =>` 分支)

**C++版本**:
```cpp
else {
    if (pitch < 114.0f) return "low_pitch";
    else if (pitch < 130.0f) return "medium_pitch";
    else if (pitch < 151.0f) return "high_pitch";
    else return "very_high_pitch";
}
```

**Rust版本**:
```rust
_ => {
    // 默认男性分类
    if pitch < 130.0 {
        "low_pitch".to_string()
    } else if pitch < 180.0 {
        "medium_pitch".to_string()
    } else if pitch < 220.0 {
        "high_pitch".to_string()
    } else {
        "very_high_pitch".to_string()
    }
}
```

**差异分析**:
- C++版本的男性默认分类使用更细致的阈值：114.0, 130.0, 151.0
- Rust版本错误地使用了未知性别的通用分类阈值：130.0, 180.0, 220.0

## 修复建议

需要修复Rust版本中男性默认分类的阈值，使其与C++版本保持一致：

```rust
_ => {
    // 默认男性分类 - 修复为与C++版本一致
    if pitch < 114.0 {
        "low_pitch".to_string()
    } else if pitch < 130.0 {
        "medium_pitch".to_string()
    } else if pitch < 151.0 {
        "high_pitch".to_string()
    } else {
        "very_high_pitch".to_string()
    }
}
```

## 总结

- **总体一致性**: 95%以上的分类规则完全一致
- **发现差异**: 1个关键差异（男性默认分类）
- **影响范围**: 影响未明确年龄段的男性音高分类
- **修复优先级**: 高（影响分类准确性）

所有女性分类、明确年龄段的男性分类、以及未知性别分类都与C++版本完全一致。只需修复男性默认分类即可实现100%一致性。