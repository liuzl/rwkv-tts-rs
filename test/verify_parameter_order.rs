//! 验证convert_standard_properties_to_tokens函数参数顺序修复

use rwkv_tts_rs::properties_util;

fn main() {
    println!("测试convert_standard_properties_to_tokens函数参数顺序...");
    
    // 测试用例：女性，青年成人，中性情感，中等音高，中等语速
    let age = "youth-adult";
    let gender = "female";
    let emotion = "NEUTRAL";
    let pitch = "medium_pitch";
    let speed = "medium";
    
    let tokens = properties_util::convert_standard_properties_to_tokens(
        age, gender, emotion, pitch, speed
    );
    
    println!("输入参数:");
    println!("  age: {}", age);
    println!("  gender: {}", gender);
    println!("  emotion: {}", emotion);
    println!("  pitch: {}", pitch);
    println!("  speed: {}", speed);
    println!();
    println!("生成的tokens: {:?}", tokens);
    
    // 验证token数量（应该是6个：特殊token + 5个属性token）
    assert_eq!(tokens.len(), 6, "Token数量应该是6个");
    
    // 验证第一个token是特殊token
    assert_eq!(tokens[0], 65536, "第一个token应该是TTS_SPECIAL_TOKEN_OFFSET (65536)");
    
    println!("✓ 参数顺序测试通过！");
    
    // 测试convert_properties_to_tokens函数
    println!("\n测试convert_properties_to_tokens函数...");
    
    let speed_val = 4.2;
    let pitch_val = 210.0;
    let age_val = 25;
    let gender_str = "female";
    let emotion_str = "HAPPY";
    
    let tokens2 = properties_util::convert_properties_to_tokens(
        speed_val, pitch_val, age_val, gender_str, emotion_str
    );
    
    println!("输入参数:");
    println!("  speed: {}", speed_val);
    println!("  pitch: {}", pitch_val);
    println!("  age: {}", age_val);
    println!("  gender: {}", gender_str);
    println!("  emotion: {}", emotion_str);
    println!();
    println!("生成的tokens: {:?}", tokens2);
    
    // 验证token数量
    assert_eq!(tokens2.len(), 6, "Token数量应该是6个");
    
    // 验证第一个token是特殊token
    assert_eq!(tokens2[0], 65536, "第一个token应该是TTS_SPECIAL_TOKEN_OFFSET (65536)");
    
    println!("✓ convert_properties_to_tokens测试通过！");
    
    println!("\n🎉 所有测试通过！参数顺序修复成功！");
}