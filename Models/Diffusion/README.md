# Forward Process of Diffusion Models

The forward process in diffusion models is a key component of how these models work. It involves gradually adding noise to the data over a series of time steps, transforming the data distribution into a standard Gaussian distribution. This process is mathematically defined and is crucial for training the model to later reverse this process to generate new data.

## Steps of the Forward Process

### 1. **Initial Data**
Start with the original data sample $ x_0 $, which could be an image, text, or any other type of data. The goal is to iteratively corrupt this data by adding noise.

### 2. **Noise Addition**
At each time step $ t $, Gaussian noise is added to the data. The process is defined as:

$$
x_t = \sqrt{\alpha_t} \cdot x_{t-1} + \sqrt{1 - \alpha_t} \cdot \epsilon,
$$

where:
- $ x_t $ : The noisy data at time step $ t $.
- $ \alpha_t $: A noise scheduling parameter that controls how much noise is added at each step.
- $ \epsilon \sim \mathcal{N}(0, I) $: Gaussian noise sampled from a normal distribution with mean 0 and identity covariance matrix $ I $.

This equation ensures that the data becomes progressively noisier as $ t $ increases. The parameter $ \alpha_t $ is typically chosen to decrease over time, allowing the noise to dominate in later steps.

## 3. **Accumulated Formula**
Instead of iteratively applying noise at each step, we can directly compute the noisy data $x_t$ at any time step $t$ using the following accumulated formula:

$$
x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon,
$$

where:
- $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$: The cumulative product of the noise scheduling parameters up to time $t$.
- $x_0$: The original data sample.
- $\epsilon \sim \mathcal{N}(0, I)$: Gaussian noise.

#### Explanation:
- $\sqrt{\bar{\alpha}_t}$ scales the original data $x_0$, ensuring that its contribution decreases as $t$ increases.
- $\sqrt{1 - \bar{\alpha}_t}$ scales the noise $\epsilon$, ensuring that the noise contribution increases as $t$ increases.
- This formula allows us to directly sample $x_t$ at any time step $t$ without iteratively computing all previous steps.

### 4. **Final Step**
After $ T $ steps, the data $ x_T $ is transformed into pure Gaussian noise:

$$
x_T \sim \mathcal{N}(0, I).
$$

At this point, the original structure of the data is completely destroyed, and the data resembles random noise.

---

## Mathematical Representation of the Forward Process

The forward process can also be expressed as a series of conditional probabilities:

$$
q(x_1, x_2, \dots, x_T \mid x_0) = \prod_{t=1}^T q(x_t \mid x_{t-1}),
$$

where each conditional probability $q(x_t \mid x_{t-1})$ represents the addition of Gaussian noise at step $t$.

Additionally, the marginal distribution of $x_t$ given $x_0$ can be written as:

$$
q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} \cdot x_0, (1 - \bar{\alpha}_t) \cdot I),
$$

where:
- $\sqrt{\bar{\alpha}_t} \cdot x_0$: The mean of the distribution, which scales the original data.
- $(1 - \bar{\alpha}_t) \cdot I$: The variance of the Gaussian noise added to the data.

This equation shows that $x_t$ is sampled from a Gaussian distribution centered around a scaled version of $x_0$, with variance increasing as $t$ progresses.

---

## Intuition Behind the Forward Process

The forward process can be thought of as "destroying" the structure of the data by adding noise step by step. This gradual corruption makes it easier for the model to learn the reverse process, where the noise is removed step by step to reconstruct the original data.

### Why Add Noise Gradually?
- **Stability**: Gradual noise addition ensures that the model can learn to reverse the process incrementally, rather than trying to denoise heavily corrupted data in one step.
- **Smooth Transition**: The gradual transition from data to noise allows the reverse process to be modeled as a series of small, predictable steps.

---

## Noise Scheduling ($ \alpha_t $)

The parameter $ \alpha_t $ plays a crucial role in controlling the amount of noise added at each step. Common schedules include:
- **Linear Schedule**: Noise increases linearly over time.
- **Cosine Schedule**: Noise increases following a cosine function for smoother transitions.
- **Exponential Schedule**: Noise increases exponentially for faster corruption.

The choice of schedule affects the quality of the generated samples and the stability of the training process.

---

# Backward Process of Diffusion Models

The **backward process** in diffusion models is the reverse of the forward process. It involves gradually removing noise from a noisy sample (starting from pure Gaussian noise) to reconstruct the original data. This process is learned during training and is crucial for generating new samples from the model.

---

## Steps of the Backward Process

### 1. **Starting Point**
The backward process begins with a noisy sample $ x_T $, which is sampled from a standard Gaussian distribution:

$$
x_T \sim \mathcal{N}(0, I).
$$

This noisy sample represents the endpoint of the forward process, where the original data structure has been completely destroyed.

---

### 2. **Reverse Transition**
At each time step $ t $, the model predicts the noise component $ \epsilon $ in the noisy data $ x_t $. Using this prediction, the model computes the denoised sample $ x_{t-1} $ for the previous time step. The reverse process is defined as:

$$
p(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t)),
$$

where:
- $ \mu_\theta(x_t, t) $: The predicted mean of the distribution, parameterized by the model.
- $ \Sigma_\theta(x_t, t) $: The predicted variance of the distribution, parameterized by the model.

The model learns $ \mu_\theta $ and $ \Sigma_\theta $ during training.

---

### 3. **Mean Prediction**
The mean $ \mu_\theta(x_t, t) $ is computed using the predicted noise $ \epsilon_\theta(x_t, t) $ and the accumulated noise schedule $ \bar{\alpha}_t $:

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \cdot \epsilon_\theta(x_t, t) \right).
$$

#### Explanation:
- $ x_t $: The noisy sample at time step $ t $.
- $ \epsilon_\theta(x_t, t) $: The predicted noise component by the model.
- $ \alpha_t $: The noise scheduling parameter at time step $ t $.
- $ \bar{\alpha}_t $: The cumulative product of noise scheduling parameters up to time $ t $.

This equation ensures that the model removes the predicted noise from $ x_t $ to compute $ x_{t-1} $.

---

### 4. **Variance Prediction**
The variance $ \Sigma_\theta(x_t, t) $ is typically fixed or learned during training. Common choices include:
- A fixed schedule based on the forward process.
- A learned parameter that adapts to the data.

---

### 5. **Iterative Denoising**
The backward process iteratively denoises the sample $ x_t $ over $ T $ steps, starting from $ x_T $ and ending at $ x_0 $, which is the reconstructed data:

$$
x_T \to x_{T-1} \to \dots \to x_1 \to x_0.
$$

At each step, the model predicts the noise and removes it to compute the sample for the previous time step.

---

## Mathematical Representation of the Backward Process

The backward process can also be expressed as a series of conditional probabilities:

$$
p(x_0, x_1, \dots, x_T) = p(x_T) \prod_{t=1}^T p(x_{t-1} \mid x_t),
$$

where:
- $ p(x_T) = \mathcal{N}(x_T; 0, I) $: The initial Gaussian noise distribution.
- $ p(x_{t-1} \mid x_t) $: The reverse transition distribution, learned by the model.

---

## Summary of the Backward Process

1. **Starting Point**: Begin with a noisy sample $ x_T \sim \mathcal{N}(0, I) $.
2. **Reverse Transition**: Iteratively compute $ x_{t-1} $ from $ x_t $ using the learned reverse distribution $ p(x_{t-1} \mid x_t) $.
3. **Noise Prediction**: Use the model to predict the noise $ \epsilon_\theta(x_t, t) $ and compute the mean $ \mu_\theta(x_t, t) $.
4. **Iterative Denoising**: Gradually remove noise over $ T $ steps to reconstruct the original data $ x_0 $.
5. **Training**: Train the model by minimizing the difference between the predicted noise and the true noise.

The backward process is the core of diffusion models, enabling them to generate high-quality samples by reversing the noise corruption applied during the forward process.



