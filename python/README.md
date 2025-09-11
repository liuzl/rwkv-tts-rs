This directory contains a compact runnable script to run RWKVTTS system.

# Instruction
0. Pre installation：python 3.11 and download all files in this directory: https://huggingface.co/yueyulin/respark/blob/main/rwkvtts-respark-webrwkv/
1. Install dependencies:
pip3 install -r requirments.txt
pip3 install webrwkv_py-0.1.0-cp311-cp311-manylinux_2_35_x86_64.whl #for linux
or
pip3 install  webrwkv_py-0.1.0-cp39-cp39-macosx_11_0_arm64.whl #for OS X

3. Run the tts_cli.py
python tts_cli.py --model_path .

# Features:
1. Chinese/English bilingual generation.
2. WebGPU based inference engine(WebRWKV) and Onnx runtime is the only environment you need. No pytorch,no tensorflow. Supporting multiple OSes(*Linux and osX are tested, other OS should work in theory)
3. Controlled voice generation using age,sex,pitch,speed,emotion.
4. Zero clone voice generation.
5. Runtime pronounciation fix using IPA and Pinyin inpainting.
# Screenshots
1. Lanuch the cli and select graphic card which is to be used:
   
![image/png](https://cdn-uploads.huggingface.co/production/uploads/63a00aa29f1f2baab2034cf8/5AM2yCcqSb6gzaxQIQBAi.png)


![image/png](https://cdn-uploads.huggingface.co/production/uploads/63a00aa29f1f2baab2034cf8/5Xq1OjcLqry_UHBx279Gs.png)

Usually the first one works best.

2. Controllable voice generation:
	2.1 Controlled voice generation:
   ![image/png](https://cdn-uploads.huggingface.co/production/uploads/63a00aa29f1f2baab2034cf8/COZ1_5vGy6fFStnJ48jV2.png)
    2.2 Input text and select output dir:
 ![image/png](https://cdn-uploads.huggingface.co/production/uploads/63a00aa29f1f2baab2034cf8/v6MahXlh-P_OaVh9A_mbb.png)
    2.3 Select properties:
   

![image/png](https://cdn-uploads.huggingface.co/production/uploads/63a00aa29f1f2baab2034cf8/sj6PhKbBICFRh8zBw5PSg.png)

![image/png](https://cdn-uploads.huggingface.co/production/uploads/63a00aa29f1f2baab2034cf8/zm8pQDo2C82yybJDtrlxU.png)

![image/png](https://cdn-uploads.huggingface.co/production/uploads/63a00aa29f1f2baab2034cf8/1jpa5DheucVQcQmG2H2zW.png)
    2.4 Finish generation and output the statistics:
![image/png](https://cdn-uploads.huggingface.co/production/uploads/63a00aa29f1f2baab2034cf8/CHbFW_o2o9E8xHTCBCfhY.png)

3. Zero clone TTS:
   3.1 Select voice wave file to clone:
![image/png](https://cdn-uploads.huggingface.co/production/uploads/63a00aa29f1f2baab2034cf8/jmd-cRd1YMm_Oj7Ea4tDa.png)
   3.2 Input the cloned voice text for exact generation:
![image/png](https://cdn-uploads.huggingface.co/production/uploads/63a00aa29f1f2baab2034cf8/LUIoKiPgnD2d7cuEfsIgr.png)


4. Pronounciation hotfix

The syntax is "SPCT_48ANY_WORD_TO_READSPCT_49IPA_OR_PINYINSPCT_50", please be careful there is no space between these SPCT_NUM.

  4.1 Chinese Pinyin:
![image/png](https://cdn-uploads.huggingface.co/production/uploads/63a00aa29f1f2baab2034cf8/srsA2QS0dsUFCjZKM-wcu.png)
    The generated file is : https://huggingface.co/yueyulin/respark/blob/main/rwkvtts-respark-webrwkv/generated_audio/%E5%B0%8F%E6%9D%8E%E8%BF%99.wav

  4.2 IPA for English:

![image/png](https://cdn-uploads.huggingface.co/production/uploads/63a00aa29f1f2baab2034cf8/sg7lyyKQJB7gArVOFYNqT.png)

   The generated file is : https://huggingface.co/yueyulin/respark/blob/main/rwkvtts-respark-webrwkv/generated_audio/SPC_4.wav
  
   The IPA is generated using library eng_to_ipa, please refer the characters this library is using.