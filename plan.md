
# 1. Overall System Structure

You are not building a generic NN.

You are building a **structured neural network that mirrors thermodynamic equations**.

Core idea:

* Each layer = one physical equation
* Each weight = thermodynamic parameter
* No arbitrary dense layers

Pipeline:

Input → Parameter blocks → Structured transformations → Activity coefficients → VLE outputs

---

# 2. Inputs and Outputs

### Inputs

* Liquid composition vector:
  x ∈ ℝⁿ
* Temperature:
  T

Optional transformed input:

* t = 1000 / T   (page 3, Eq. 82)

---

### Outputs

* Activity coefficients: γᵢ
* Optional:

  * Vapor composition yᵢ
  * Pressure P

---

# 3. Parameter Blocks (Trainable + Fixed)

You must separate parameters clearly.

### Trainable parameters

From paper (page 2):

* a = βₐ[s₁]
* A = β_A[s₁]
* B = β_B[s₃]  

These correspond to:

* temperature polynomial coefficients
* interaction parameters

---

### Fixed matrices (κ matrices)

From Table 1 (page 1):

* κ₁ to κ₁₀ define structured connectivity
* These are NOT learned
* They enforce thermodynamic structure

Examples:

* identity expansions
* stacking matrices
* repetition operators

---

# 4. Core Architecture Blocks

Break the network into **5 main modules**

---

## Module 1: Temperature Parameterization

Goal:
Convert T into τᵢⱼ

From page 2:

* B_T = (κ₁ ⊙ (x ⊗ B))
* τ = κ₂ A + κ₃ B_T  

Architecture:

* Input: T
* Apply polynomial expansion
* Output: τ matrix (n × n)

---

## Module 2: Non-randomness Factor

From page 2:

* W = exp(-κ₂ ⊙ κ₅ τ)

This is critical.

Architecture:

* Elementwise exponential layer
* No activation freedom, fixed function

---

## Module 3: G and V Matrices

From page 2:

* V = κ₅ τ ⊙ κ₅ W

This creates:

* weighted interaction terms

---

## Module 4: Structured Layer Stack (L1 → L12)

This is the **core ASNN pipeline**

From page 2 (Eqs 36–48):

You must implement sequential layers:

* L1 = κ₆ x ⊙ κ₅ V
* L2 = κ₇ L1
* L3 = κ₈ x ⊙ κ₅ V
* L4 = κ₇ L3
* L5 = 1 ⊘ (κ₅ L2)
* L6 = f((κ₅ L4) ⊙ (κ₅ L5))
* L7 = κ₅ L6
* L8 = (κ₁₀ V) ⊙ (κ₄ L4)
* L9 = (κ₁₀ V) ⊙ (κ₄ L2)
* L10 = −κ₅ L8 + κ₅ L9
* L11 = f((κ₅ L4) ⊙ (κ₅ L5))
* L12 = combination of κ matrices and previous layers  

Activation:

* f = tanh (as per paper)

Important:

* Every operation is **elementwise or structured matrix multiplication**
* No dense layers

---

## Module 5: Activity Coefficient Output

From page 1 (Eq. 27):

* ln(γᵢ) computed from:

  * weighted sums of V and W
  * ratio structures

Architecture:

* Final aggregation layer
* Output:
  ln(γ)
* Then:
  γ = exp(output)

---

# 5. VLE Extension Module

From page 3:

Add:

### Raoult’s law

* yᵢ P = xᵢ γᵢ pᵢˢᵃᵗ

### Antoine equation

* pᵢˢᵃᵗ = exp(K₁ + K₂/(T+K₃) + K₄T + K₅ ln T + K₆ T^K₇)  

### Final outputs

* y
* P

---

# 6. Loss Function

From page 3:

* L = Σ ((Y_exp − Y_calc)/Y_exp)²  

Where:

* Y = γ, y, or P

---

# 7. Training Strategy

### Data split

* 80% training
* 10% validation
* 10% test  

---

### Constraints

* Enforce:

  * physical bounds on τ
  * positivity of γ

---

### Optimization

* Use gradient-based optimizer
* Only update:

  * β parameters

---

# 8. Python Architecture (High-Level)

Organize like this:

### 1. data/

* dataset loader
* preprocessing

### 2. parameters/

* β parameter class
* κ matrix generator

### 3. layers/

Each equation = one class

* TauLayer
* WLayer
* VLayer
* L1Layer … L12Layer

### 4. model/

* NRTL_ASNN class
* forward pass = sequential execution of layers

### 5. physics/

* Antoine equation
* VLE relations

### 6. training/

* loss
* optimizer
* training loop

---

# 9. Key Design Rules

Follow strictly:

* No random dense layers
* No arbitrary architecture tuning
* Every node must correspond to an equation
* Keep matrix dimensions consistent with κ definitions
* Preserve interpretability

---

# 10. Mental Model

Think of this as:

* Not a neural network
* A **differentiable thermodynamic solver**

---


