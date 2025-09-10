//! 属性处理工具模块
//! 实现与Python版本properties_util.py相同的功能

/// 速度映射表
const SPEED_MAP: &[(&str, &str)] = &[
    ("very_slow", "SPCT_1"),
    ("slow", "SPCT_2"),
    ("medium", "SPCT_3"),
    ("fast", "SPCT_4"),
    ("very_fast", "SPCT_5"),
];

/// 音高映射表
const PITCH_MAP: &[(&str, &str)] = &[
    ("low_pitch", "SPCT_6"),
    ("medium_pitch", "SPCT_7"),
    ("high_pitch", "SPCT_8"),
    ("very_high_pitch", "SPCT_9"),
];

/// 年龄映射表
const AGE_MAP: &[(&str, &str)] = &[
    ("child", "SPCT_13"),
    ("teenager", "SPCT_14"),
    ("youth-adult", "SPCT_15"),
    ("middle-aged", "SPCT_16"),
    ("elderly", "SPCT_17"),
];

/// 性别映射表
const GENDER_MAP: &[(&str, &str)] = &[("female", "SPCT_46"), ("male", "SPCT_47")];

/// 情感映射表
const EMOTION_MAP: &[(&str, &str)] = &[
    ("UNKNOWN", "SPCT_21"),
    ("NEUTRAL", "SPCT_22"),
    ("ANGRY", "SPCT_23"),
    ("HAPPY", "SPCT_24"),
    ("SAD", "SPCT_25"),
    ("FEARFUL", "SPCT_26"),
    ("DISGUSTED", "SPCT_27"),
    ("SURPRISED", "SPCT_28"),
    ("SARCASTIC", "SPCT_29"),
    ("EXCITED", "SPCT_30"),
    ("SLEEPY", "SPCT_31"),
    ("CONFUSED", "SPCT_32"),
    ("EMPHASIS", "SPCT_33"),
    ("LAUGHING", "SPCT_34"),
    ("SINGING", "SPCT_35"),
    ("WORRIED", "SPCT_36"),
    ("WHISPER", "SPCT_37"),
    ("ANXIOUS", "SPCT_38"),
    ("NO-AGREEMENT", "SPCT_39"),
    ("APOLOGETIC", "SPCT_40"),
    ("CONCERNED", "SPCT_41"),
    ("ENUNCIATED", "SPCT_42"),
    ("ASSERTIVE", "SPCT_43"),
    ("ENCOURAGING", "SPCT_44"),
    ("CONTEMPT", "SPCT_45"),
];

/// 将标准属性转换为tokens
///
/// # Arguments
/// * `age` - 年龄属性
/// * `gender` - 性别属性
/// * `emotion` - 情感属性
/// * `pitch` - 音高属性
/// * `speed` - 语速属性
///
/// # Returns
/// * `String` - 属性tokens字符串
pub fn convert_standard_properties_to_tokens(
    age: &str,
    gender: &str,
    emotion: &str,
    pitch: &str,
    speed: &str,
) -> String {
    let age_token = get_token_from_map(AGE_MAP, age);
    let gender_token = get_token_from_map(GENDER_MAP, gender);
    let emotion_token = get_token_from_map(EMOTION_MAP, emotion);
    let pitch_token = get_token_from_map(PITCH_MAP, pitch);
    let speed_token = get_token_from_map(SPEED_MAP, speed);

    format!(
        "SPCT_0{}{}{}{}{}",
        age_token, gender_token, emotion_token, pitch_token, speed_token
    )
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
pub fn classify_pitch(pitch: f32, gender: &str, age: &str) -> String {
    let gender = gender.to_lowercase();
    let age = age.to_lowercase();

    // 女性分类
    if gender == "female" {
        match age.as_str() {
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
        match age.as_str() {
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

/// 根据音高和语速值转换属性为tokens
///
/// # Arguments
/// * `age` - 年龄属性
/// * `gender` - 性别属性
/// * `emotion` - 情感属性
/// * `pitch` - 音高值
/// * `speed` - 语速值
///
/// # Returns
/// * `String` - 属性tokens字符串
pub fn convert_properties_to_tokens(
    age: &str,
    gender: &str,
    emotion: &str,
    pitch: f32,
    speed: f32,
) -> String {
    let age_token = get_token_from_map(AGE_MAP, age);
    let gender_token = get_token_from_map(GENDER_MAP, gender);
    let emotion_token = get_token_from_map(EMOTION_MAP, emotion);
    let pitch_token = get_token_from_map(PITCH_MAP, &classify_pitch(pitch, gender, age));
    let speed_token = get_token_from_map(SPEED_MAP, &classify_speed(speed));

    format!(
        "SPCT_0{}{}{}{}{}",
        age_token, gender_token, emotion_token, pitch_token, speed_token
    )
}

/// 从映射表中获取token
fn get_token_from_map(map: &[(&str, &str)], key: &str) -> String {
    for &(k, v) in map {
        if k.eq_ignore_ascii_case(key) {
            return v.to_string();
        }
    }
    // 如果找不到匹配项，返回默认值
    map[0].1.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_standard_properties_to_tokens() {
        let result = convert_standard_properties_to_tokens(
            "youth-adult",
            "female",
            "NEUTRAL",
            "medium_pitch",
            "medium",
        );
        assert_eq!(result, "SPCT_0SPCT_15SPCT_46SPCT_22SPCT_7SPCT_3");
    }

    #[test]
    fn test_classify_pitch() {
        let result = classify_pitch(200.0, "female", "youth-adult");
        assert_eq!(result, "medium_pitch");
    }

    #[test]
    fn test_classify_speed() {
        let result = classify_speed(4.2);
        assert_eq!(result, "medium");
    }

    #[test]
    fn test_convert_properties_to_tokens() {
        let result = convert_properties_to_tokens("youth-adult", "female", "NEUTRAL", 200.0, 4.2);
        assert_eq!(result, "SPCT_0SPCT_15SPCT_46SPCT_22SPCT_7SPCT_3");
    }
}
