# Training Pipeline — Step by Step

## Step 1: Prepare the Agriculture-Aware LLM
| Component | Details |
|-----------|---------|
| **Train** | LoRA on base LLM |
| **Data** | Agriculture text corpus |
| **Loss** | Causal LM (next token prediction) |
| **Result** | LLM that knows plant diseases, agronomic terminology |

## Step 2: Train the Visual Projector
| Component | Details |
|-----------|---------|
| **Freeze** | ViT + LLM |
| **Train** | MLP projector only |
| **Data** | (image, caption) pairs from disease dataset |
| **Loss** | Causal LM on captions given image tokens |
| **Result** | Projector that speaks the LLM's embedding language |
| **Purpose** | Alignment stage — teach image tokens to "look like" text tokens |

## Step 3: Fine-tune End-to-End with LoRA
| Component | Details |
|-----------|---------|
| **Freeze** | ViT |
| **Train** | MLP projector + new LoRA layers in LLM |
| **Data** | (image, question, answer) triplets — disease VQA |
| **Loss** | Causal LM on answers |
| **Result** | Full multimodal agriculture assistant |