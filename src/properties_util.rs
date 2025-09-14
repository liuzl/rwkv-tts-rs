//! 属性处理工具模块
//! 实现与Python版本properties_util.py相同的功能

/// TTS特殊token偏移量，对应C++中的tts_special_token_offset
const TTS_SPECIAL_TOKEN_OFFSET: i32 = 77823;

/// 速度映射表
const SPEED_MAP: &[(&str, i32)] = &[
    ("very_slow", 1),
    ("slow", 2),
    ("medium", 3),
    ("fast", 4),
    ("very_fast", 5),
];

/// 音高映射表
const PITCH_MAP: &[(&str, i32)] = &[
    ("low_pitch", 6),
    ("medium_pitch", 7),
    ("high_pitch", 8),
    ("very_high_pitch", 9),
];

/// 年龄映射表
const AGE_MAP: &[(&str, i32)] = &[
    ("child", 13),
    ("teenager", 14),
    ("youth-adult", 15),
    ("middle-aged", 16),
    ("elderly", 17),
];

/// 性别映射表
const GENDER_MAP: &[(&str, i32)] = &[("female", 46), ("male", 47)];

/// 情感映射表
const EMOTION_MAP: &[(&str, i32)] = &[
    ("UNKNOWN", 21),
    ("ANGRY", 22),
    ("DISGUSTED", 23),
    ("FEARFUL", 24),
    ("HAPPY", 25),
    ("NEUTRAL", 26),
    ("SAD", 27),
    ("SURPRISED", 28),
    ("ANNOYED", 29),
    ("TIRED", 30),
    ("LAUGHING", 31),
    ("TERRIFIED", 32),
    ("SHOUTING", 33),
    ("WHISPERING", 34),
    ("UNFRIENDLY", 35),
    ("ENUNCIATED", 36),
    ("SINGING", 37),
    ("QUESTIONING", 38),
    ("CONFUSED", 39),
    ("SERIOUS", 40),
    ("SMILING", 41),
    ("EXCITED", 42),
    ("FRIENDLY", 43),
    ("HUMOROUS", 44),
    ("CONTEMPT", 45),
];

/// 将标准属性转换为token ID数组
///
/// # 参数
/// * `speed` - 语速 ("very_slow", "slow", "medium", "fast", "very_fast")
/// * `pitch` - 音高 ("low_pitch", "medium_pitch", "high_pitch", "very_high_pitch")
/// * `age` - 年龄 ("child", "teenager", "youth-adult", "middle-aged", "elderly")
/// * `gender` - 性别 ("female", "male")
/// * `emotion` - 情感 (见EMOTION_MAP)
///
/// # 返回值
/// 返回token ID数组，第一个是TTS_SPECIAL_TOKEN_OFFSET，后续是各属性对应的token ID
pub fn convert_standard_properties_to_tokens(
    speed: &str,
    pitch: &str,
    age: &str,
    gender: &str,
    emotion: &str,
) -> Vec<i32> {
    let speed_token = get_token_from_map(SPEED_MAP, speed).unwrap_or(3);
    let pitch_token = get_token_from_map(PITCH_MAP, pitch).unwrap_or(7);
    let age_token = get_token_from_map(AGE_MAP, age).unwrap_or(15);
    let gender_token = get_token_from_map(GENDER_MAP, gender).unwrap_or(46);
    let emotion_token = get_token_from_map(EMOTION_MAP, emotion).unwrap_or(26);

    vec![
        TTS_SPECIAL_TOKEN_OFFSET,
        TTS_SPECIAL_TOKEN_OFFSET + speed_token,
        TTS_SPECIAL_TOKEN_OFFSET + pitch_token,
        TTS_SPECIAL_TOKEN_OFFSET + age_token,
        TTS_SPECIAL_TOKEN_OFFSET + gender_token,
        TTS_SPECIAL_TOKEN_OFFSET + emotion_token,
    ]
}

/// 根据音高值分类
///
/// # Arguments
/// * `pitch` - 音高值
/// * `gender` - 性别
/// * `age` - 年龄
///
/// # Returns
/// * `String` - 分类后的音高属性
pub fn classify_pitch(pitch: f32, gender: &str, age: u8) -> String {
    let gender = gender.to_lowercase();
    let age_class = classify_age(age);

    // 女性分类
    if gender == "female" {
        match age_class.as_str() {
            "child" => {
                if pitch < 250.0 {
                    "low_pitch".to_string()
                } else if pitch < 290.0 {
                    "medium_pitch".to_string()
                } else {
                    "high_pitch".to_string()
                }
            }
            "teenager" => {
                if pitch < 208.0 {
                    "low_pitch".to_string()
                } else if pitch < 238.0 {
                    "medium_pitch".to_string()
                } else if pitch < 270.0 {
                    "high_pitch".to_string()
                } else {
                    "very_high_pitch".to_string()
                }
            }
            "youth-adult" => {
                if pitch < 191.0 {
                    "low_pitch".to_string()
                } else if pitch < 211.0 {
                    "medium_pitch".to_string()
                } else if pitch < 232.0 {
                    "high_pitch".to_string()
                } else {
                    "very_high_pitch".to_string()
                }
            }
            "middle-aged" => {
                if pitch < 176.0 {
                    "low_pitch".to_string()
                } else if pitch < 195.0 {
                    "medium_pitch".to_string()
                } else if pitch < 215.0 {
                    "high_pitch".to_string()
                } else {
                    "very_high_pitch".to_string()
                }
            }
            "elderly" => {
                if pitch < 170.0 {
                    "low_pitch".to_string()
                } else if pitch < 190.0 {
                    "medium_pitch".to_string()
                } else if pitch < 213.0 {
                    "high_pitch".to_string()
                } else {
                    "very_high_pitch".to_string()
                }
            }
            _ => {
                // 默认女性分类
                if pitch < 187.0 {
                    "low_pitch".to_string()
                } else if pitch < 209.0 {
                    "medium_pitch".to_string()
                } else if pitch < 232.0 {
                    "high_pitch".to_string()
                } else {
                    "very_high_pitch".to_string()
                }
            }
        }
    }
    // 男性分类
    else if gender == "male" {
        match age_class.as_str() {
            "teenager" => {
                if pitch < 121.0 {
                    "low_pitch".to_string()
                } else if pitch < 143.0 {
                    "medium_pitch".to_string()
                } else if pitch < 166.0 {
                    "high_pitch".to_string()
                } else {
                    "very_high_pitch".to_string()
                }
            }
            "youth-adult" => {
                if pitch < 115.0 {
                    "low_pitch".to_string()
                } else if pitch < 131.0 {
                    "medium_pitch".to_string()
                } else if pitch < 153.0 {
                    "high_pitch".to_string()
                } else {
                    "very_high_pitch".to_string()
                }
            }
            "middle-aged" => {
                if pitch < 110.0 {
                    "low_pitch".to_string()
                } else if pitch < 125.0 {
                    "medium_pitch".to_string()
                } else if pitch < 147.0 {
                    "high_pitch".to_string()
                } else {
                    "very_high_pitch".to_string()
                }
            }
            "elderly" => {
                if pitch < 115.0 {
                    "low_pitch".to_string()
                } else if pitch < 128.0 {
                    "medium_pitch".to_string()
                } else if pitch < 142.0 {
                    "high_pitch".to_string()
                } else {
                    "very_high_pitch".to_string()
                }
            }
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
        }
    }
    // 未知性别，使用通用分类
    else if pitch < 130.0 {
        "low_pitch".to_string()
    } else if pitch < 180.0 {
        "medium_pitch".to_string()
    } else if pitch < 220.0 {
        "high_pitch".to_string()
    } else {
        "very_high_pitch".to_string()
    }
}

/// 根据语速值分类
///
/// # Arguments
/// * `speed` - 语速值
///
/// # Returns
/// * `String` - 分类后的语速属性
pub fn classify_speed(speed: f32) -> String {
    if speed <= 3.5 {
        "very_slow".to_string()
    } else if speed < 4.0 {
        "slow".to_string()
    } else if speed <= 4.5 {
        "medium".to_string()
    } else if speed <= 5.0 {
        "fast".to_string()
    } else {
        "very_fast".to_string()
    }
}

/// 根据年龄值分类年龄属性
///
/// # 参数
/// * `age` - 年龄值 (0-100)
///
/// # 返回值
/// 返回年龄分类字符串
fn classify_age(age: u8) -> String {
    if age < 13 {
        "child".to_string()
    } else if age < 20 {
        "teenager".to_string()
    } else if age < 40 {
        "youth-adult".to_string()
    } else if age < 65 {
        "middle-aged".to_string()
    } else {
        "elderly".to_string()
    }
}

/// 将属性转换为token ID数组
///
/// # 参数
/// * `speed` - 语速值 (0.0-2.0)
/// * `pitch` - 音高值 (-20.0到20.0)
/// * `age` - 年龄值 (0-100)
/// * `gender` - 性别 ("female", "male")
/// * `emotion` - 情感 (见EMOTION_MAP)
///
/// # 返回值
/// 返回token ID数组，第一个是TTS_SPECIAL_TOKEN_OFFSET，后续是各属性对应的token ID
pub fn convert_properties_to_tokens(
    speed: f32,
    pitch: f32,
    age: u8,
    gender: &str,
    emotion: &str,
) -> Vec<i32> {
    let speed_class = classify_speed(speed);
    let pitch_class = classify_pitch(pitch, gender, age);
    let age_class = classify_age(age);

    convert_standard_properties_to_tokens(&speed_class, &pitch_class, &age_class, gender, emotion)
}

/// 从映射表中获取token
fn get_token_from_map(map: &[(&str, i32)], key: &str) -> Option<i32> {
    for &(k, v) in map {
        if k.eq_ignore_ascii_case(key) {
            return Some(v);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_standard_properties_to_tokens() {
        let result = convert_standard_properties_to_tokens(
            "medium",
            "medium_pitch",
            "youth-adult",
            "female",
            "NEUTRAL",
        );
        let expected = vec![
            TTS_SPECIAL_TOKEN_OFFSET,      // 77823
            TTS_SPECIAL_TOKEN_OFFSET + 3,  // 77826 (medium speed)
            TTS_SPECIAL_TOKEN_OFFSET + 7,  // 77830 (medium_pitch)
            TTS_SPECIAL_TOKEN_OFFSET + 15, // 77838 (youth-adult)
            TTS_SPECIAL_TOKEN_OFFSET + 46, // 77869 (female)
            TTS_SPECIAL_TOKEN_OFFSET + 26, // 77849 (NEUTRAL)
        ];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_convert_properties_to_tokens() {
        let result = convert_properties_to_tokens(4.2, 200.0, 25, "female", "NEUTRAL");
        // 应该返回包含6个token ID的数组
        assert_eq!(result.len(), 6);
        assert_eq!(result[0], TTS_SPECIAL_TOKEN_OFFSET);
        // 检查所有token ID都大于等于TTS_SPECIAL_TOKEN_OFFSET
        for &token_id in &result {
            assert!(token_id >= TTS_SPECIAL_TOKEN_OFFSET);
        }
    }

    #[test]
    fn test_classify_pitch() {
        let result = classify_pitch(200.0, "female", 25);
        assert_eq!(result, "medium_pitch"); // 女性25岁(youth-adult): 191.0 < 200.0 < 211.0

        let result = classify_pitch(100.0, "male", 25);
        assert_eq!(result, "low_pitch"); // 男性25岁(youth-adult): 100.0 < 115.0

        let result = classify_pitch(200.0, "female", 25);
        assert_eq!(result, "medium_pitch"); // 女性25岁(youth-adult): 191.0 < 200.0 < 211.0
    }

    #[test]
    fn test_classify_speed() {
        let result = classify_speed(4.2);
        assert_eq!(result, "medium");
    }

    #[test]
    fn test_classify_age() {
        assert_eq!(classify_age(10), "child");
        assert_eq!(classify_age(16), "teenager");
        assert_eq!(classify_age(25), "youth-adult");
        assert_eq!(classify_age(45), "middle-aged");
        assert_eq!(classify_age(70), "elderly");
    }
}
