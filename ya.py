# https://huggingface.co/yandex/YandexGPT-5-Lite-8B-pretrain

from vllm import LLM, SamplingParams


MODEL_NAME = "yandex/YandexGPT-5-Lite-8B-pretrain" # 16.1 GB

sampling_params = SamplingParams(
    temperature=0.3,
    max_tokens=500,
)

llm = LLM(
    MODEL_NAME,
    # tensor_parallel_size=1,
    enforce_eager=True,
    trust_remote_code=True,
    gpu_memory_utilization=0.95, 
    max_model_len=11000,
)
# input_texts = ["Кто сказал тебе, что нет на свете настоящей,"] # Кто сказал тебе, что нет на свете настоящей, верной, вечной любви? Да отрежут лгуну его гнусный язык!
# outputs = llm.generate(input_texts, use_tqdm=False, sampling_params=sampling_params)

# for i in range(len(input_texts)):
#     print(input_texts[i] + outputs[i].outputs[0].text)

messages = [
            # "What about solving an (1.5*x + 1)^9 = 512 equation?"
            # "What about solving an 2x + 3 = 7 equation?"
            # "What about solving an x*x + 6*x + 9 = 16 equation?"
            "What about solving an x*x*x + 3*x*x + 3*x + 1 = 125 equation?"
            ]



output = llm.generate(messages, use_tqdm=False, sampling_params=sampling_params)
print(output[0].outputs[0].text)

# Answer: 

# The solution to the equation (1.5x + 1)^9 = 512 is x = 4.
#  To solve the equation (1.5x + 1)^9 = 512, we need to find the value of x that satisfies the equation.
#  First, we can take the ninth root of both sides of the equation to eliminate the exponent: 
# (1.5x + 1)^(1/9) = 512^(1/9)
#  Simplifying the right-hand side, we get: 
# (1.5x + 1)^(1/9) = 2
#  Next, we can raise both sides of the equation to the ninth power to eliminate the exponent: 
# (1.5x + 1) = 2^9
#  Simplifying the right-hand side, we get: 
# (1.5x + 1) = 512
#  Subtracting 1 from both sides, we get: 
# 1.5x = 511
#  Finally, we can divide both sides by 1.5 to solve for x: 
# x = 511/1.5
# x = 340.6667
#  Therefore, the solution to the equation (1.5x + 1)^9 = 512 is x = 340.6667.



# Answer: 

# 2x + 3 = 7
# 2x = 7 - 3
# 2x = 4
# x = 4/2
# x = 2
#  Hope this helps!


# Answer: 

# x*x + 6*x + 9 = 16
# x^2 + 6x + 9 = 16
# x^2 + 6x + 9 - 16 = 0
# x^2 + 6x - 7 = 0
# (x + 7)(x - 1) = 0
# x = -7 or x = 1


# Answer: 

# x*x*x + 3*x*x + 3*x + 1 = 125
# x³ + 3x² + 3x + 1 = 125
# x³ + 3x² + 3x + 1 - 125 = 0
# x³ + 3x² + 3x - 124 = 0
# x³ + 3x² + 3x = 124
# x(x² + 3x + 3) = 124
# x = 124/(x² + 3x + 3)
# x = 124/[(x + 1)² + 2]
# x = 124/[(x + 1)² + 2]
# x = 124/[(x + 1)² + 2]
# x = 124/[(x + 1)² + 2]
# x = 124/[(x + 1)² + 2]
# x = 124/[(x + 1)² + 2]
# x = 124/[(x + 1)² + 2]
# x = 124/[(x + 1)² + 2]
# x = 124/[(x + 1)² + 2]
# x = 124/[(x + 1)² + 2]
# x = 124/[(x + 1)² + 2]
# x = 124/[(x + 1)² + 2]
# x = 124/[(x + 1)² + 2]
# x = 124/[(x + 1)² + 2]
# x = 124/[(x + 1)² + 2]
# x = 124/[(x + 1)² + 2]
# x = 124/[(x + 1)² + 2]
# x = 124/[(x + 1)² + 2]
# x = 124/[(x
