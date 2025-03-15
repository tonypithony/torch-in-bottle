# pip install torch torchvision torchaudio
# pip install vllm
# pip install -r requirements.txt

# https://huggingface.co/microsoft/Phi-4-mini-instruct
# https://pypi.org/project/vllm/
# https://stackoverflow.com/questions/79439095/remove-gpu-ram-with-vllm

from vllm import LLM, SamplingParams

llm = LLM(model="microsoft/Phi-4-mini-instruct", # 7.7 GB
		  trust_remote_code=True,
		  gpu_memory_utilization=0.95, 
		  max_model_len=35000,)

messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
    {"role": """assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 
			    1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 
			    2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."""},
    
    # {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
    # {"role": "user", "content": "What about solving an x*x + 6*x + 9 = 16 equation?"},
    # {"role": "user", "content": "What about solving an x*x*x + 3*x*x + 3*x + 1 = 125 equation?"},
    # {"role": "user", "content": "What about solving an (x + 1)^9 = 512 equation?"},
    {"role": "user", "content": "What about solving an (1.5*x + 1)^9 = 512 equation?"},
]

sampling_params = SamplingParams(
  max_tokens=500,
  temperature=0.0,
)

output = llm.chat(messages=messages, sampling_params=sampling_params)
print(output[0].outputs[0].text)

'''
To solve the equation 2x + 3 = 7, follow these steps:

1. Subtract 3 from both sides of the equation to isolate the term with the variable (x):
   2x + 3 - 3 = 7 - 3
   2x = 4

2. Divide both sides of the equation by 2 to solve for x:
   2x / 2 = 4 / 2
   x = 2

So, the solution to the equation 2x + 3 = 7 is x = 2.
'''


# To solve the equation \(2x + 3 = 7\), follow these steps:

# 1. Subtract 3 from both sides of the equation to isolate the term with the variable \(x\):
#    \[
#    2x + 3 - 3 = 7 - 3
#    \]
#    Simplifying this, we get:
#    \[
#    2x = 4
#    \]

# 2. Divide both sides of the equation by 2 to solve for \(x\):
#    \[
#    \frac{2x}{2} = \frac{4}{2}
#    \]
#    Simplifying this, we get:
#    \[
#    x = 2
#    \]

# So, the solution to the equation \(2x + 3 = 7\) is \(x = 2\).

# Sure, let's solve the equation step by step.

# The equation is:
# \[ x^2 + 6x + 9 = 16 \]

# First, let's move all terms to one side of the equation to set it to zero:
# \[ x^2 + 6x + 9 - 16 = 0 \]
# \[ x^2 + 6x - 7 = 0 \]

# Now, we have a quadratic equation in the form \( ax^2 + bx + c = 0 \). We can solve this using the quadratic formula:
# \[ x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} \]

# For our equation, \( a = 1 \), \( b = 6 \), and \( c = -7 \). Plugging these values into the quadratic formula gives:
# \[ x = \frac{-6 \pm \sqrt{6^2 - 4 \cdot 1 \cdot (-7)}}{2 \cdot 1} \]
# \[ x = \frac{-6 \pm \sqrt{36 + 28}}{2} \]
# \[ x = \frac{-6 \pm \sqrt{64}}{2} \]
# \[ x = \frac{-6 \pm 8}{2} \]

# This gives us two solutions:
# \[ x = \frac{-6 + 8}{2} = \frac{2}{2} = 1 \]
# \[ x = \frac{-6 - 8}{2} = \frac{-14}{2} = -7 \]

# So, the solutions to the equation \( x^2 + 6x + 9 = 16 \) are:
# \[ x = 1 \]
# \[ x = -7 \]




# To solve the quadratic equation \( x^2 + 6x + 9 = 16 \), follow these steps:

# 1. First, move all terms to one side of the equation to set it to zero:
#    \[
#    x^2 + 6x + 9 - 16 = 0
#    \]
#    Simplify the equation:
#    \[
#    x^2 + 6x - 7 = 0
#    \]

# 2. Now, we can solve this quadratic equation using the quadratic formula, which is:
#    \[
#    x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
#    \]
#    For the equation \( x^2 + 6x - 7 = 0 \), the coefficients are \( a = 1 \), \( b = 6 \), and \( c = -7 \).

# 3. Plug these values into the quadratic formula:
#    \[
#    x = \frac{-6 \pm \sqrt{6^2 - 4 \cdot 1 \cdot (-7)}}{2 \cdot 1}
#    \]
#    Simplify inside the square root:
#    \[
#    x = \frac{-6 \pm \sqrt{36 + 28}}{2}
#    \]
#    \[
#    x = \frac{-6 \pm \sqrt{64}}{2}
#    \]
#    \[
#    x = \frac{-6 \pm 8}{2}
#    \]

# 4. This gives us two solutions:
#    \[
#    x = \frac{-6 + 8}{2} = \frac{2}{2} = 1
#    \]
#    \[
#    x = \frac{-6 - 8}{2} = \frac{-14}{2} = -7
#    \]

# So, the solutions to the equation \( x^2 + 6x + 9 = 16 \) are \( x = 1 \) and \( x = -7 \).


# To solve the equation \( x^3 + 3x^2 + 3x + 1 = 125 \), follow these steps:

# 1. Subtract 125 from both sides to set the equation to zero:
#    \[
#    x^3 + 3x^2 + 3x + 1 - 125 = 0
#    \]
#    Simplify:
#    \[
#    x^3 + 3x^2 + 3x - 124 = 0
#    \]

# 2. Look for possible rational roots using the Rational Root Theorem, which suggests that any rational solution, \( p/q \), is a factor of the constant term (-124) divided by a factor of the leading coefficient (1). The possible rational roots are \( \pm 1, \pm 2, \pm 4, \pm 31, \pm 62, \pm 124 \).

# 3. Test these possible roots by substituting them into the equation to see if they satisfy it.

#    Let's test \( x = 4 \):
#    \[
#    4^3 + 3(4^2) + 3(4) - 124 = 64 + 48 + 12 - 124 = 124 - 124 = 0
#    \]
#    So, \( x = 4 \) is a root of the equation.

# 4. Since \( x = 4 \) is a root, we can factor the polynomial as:
#    \[
#    (x - 4)(x^2 + ax + b) = x^3 + 3x^2 + 3x - 124
#    \]

# 5. To find \( a \) and \( b \), perform polynomial division or use synthetic division to divide \( x^3 + 3x^2 + 3x - 124 \) by \( x - 4 \).

#    Using synthetic division with root \( x = 4 \):
#    \[
#    \begin{array}{r|rrrr}
#      4 & 1 & 3 & 3 & -124 \\
#        &   & 4 & 28 & 124 \\
#      \hline
#        & 1 & 7 & 31 & 0 \\
#    \end{array}
#    \]
#    The quotient is \( x^2 + 7x + 31 \).


# To solve the equation (x + 1)^9 = 512, you can follow these steps:

# 1. Recognize that 512 is a power of 2, specifically 2^9.
# 2. Rewrite the equation with this knowledge: (x + 1)^9 = 2^9.
# 3. Since the bases are the same and the exponents are equal, you can set the bases equal to each other: x + 1 = 2.
# 4. Solve for x by subtracting 1 from both sides: x = 2 - 1.
# 5. Simplify the result: x = 1.

# So, the solution to the equation (x + 1)^9 = 512 is x = 1.


# To solve the equation \((x + 1)^9 = 512\), follow these steps:

# 1. Recognize that 512 is a power of 2. Specifically, \(512 = 2^9\).

# 2. Rewrite the equation using this information:
#    \[
#    (x + 1)^9 = 2^9
#    \]

# 3. Since the bases are the same, you can set the exponents equal to each other:
#    \[
#    x + 1 = 2
#    \]

# 4. Solve for \(x\):
#    \[
#    x = 2 - 1
#    \]
#    \[
#    x = 1
#    \]

# So, the solution to the equation \((x + 1)^9 = 512\) is \(x = 1\).


# To solve the equation (1.5x + 1)^9 = 512, you can follow these steps:

# 1. Recognize that 512 is a power of 2, specifically 2^9.
# 2. Rewrite the equation with this knowledge: (1.5x + 1)^9 = 2^9.
# 3. Since the bases are now the same, you can equate the exponents: 1.5x + 1 = 2.
# 4. Solve for x: 1.5x = 2 - 1, which simplifies to 1.5x = 1.
# 5. Divide both sides by 1.5 to isolate x: x = 1 / 1.5.
# 6. Simplify the fraction: x = 2/3.

# So the solution to the equation (1.5x + 1)^9 = 512 is x = 2/3.


# To solve the equation \((1.5x + 1)^9 = 512\), follow these steps:

# 1. **Isolate the base**: Start by taking the 9th root of both sides of the equation to isolate the term \((1.5x + 1)\).

#    \[
#    (1.5x + 1)^9 = 512
#    \]

#    Taking the 9th root of both sides:

#    \[
#    1.5x + 1 = \sqrt[9]{512}
#    \]

# 2. **Simplify the right-hand side**: Calculate the 9th root of 512. Note that \(512 = 2^9\), so:

#    \[
#    \sqrt[9]{512} = \sqrt[9]{2^9} = 2
#    \]

#    Now the equation is:

#    \[
#    1.5x + 1 = 2
#    \]

# 3. **Solve for \(x\)**: Subtract 1 from both sides to isolate the term with \(x\):

#    \[
#    1.5x = 2 - 1
#    \]

#    \[
#    1.5x = 1
#    \]

#    Divide both sides by 1.5:

#    \[
#    x = \frac{1}{1.5}
#    \]

#    Simplify the fraction:

#    \[
#    x = \frac{1}{1.5} = \frac{1}{\frac{3}{2}} = \frac{2}{3}
#    \]

# So, the solution to the equation \((1.5x + 1)^9 = 512\) is:

# \[
# x = \frac{2}{3}
# \]