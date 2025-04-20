  The reason models return {'fmaps': [x_1, x_2, x_3], 'features': features} instead of classifier logits is to support a modular architecture for continual learning. This design supports multiple key
  requirements:

  1. Feature Extraction vs. Classification:
    - This separation creates a clean boundary between feature extraction (done by convnet) and classification (done by fully connected layers)
    - The BaseNet and other network classes in inc_net.py handle the actual classification by adding FC layers
  2. Multi-level Feature Preservation:
    - Continual learning techniques like PODNet and other distillation-based methods need access to intermediate feature maps (not just final features)
    - These intermediate representations (x_1, x_2, x_3) help preserve knowledge when learning new tasks
  3. Knowledge Distillation:
    - Several methods (like PODNet) compare feature maps from old and new models to transfer knowledge
    - The spatial structure in these maps contains important information that would be lost if only using the final features
  4. Flexibility for Different Methods:
    - Different continual learning strategies need different types of feature information
    - Some need only final features, others need intermediate feature maps for spatial distillation
    - This unified interface supports all these approaches

  This architecture allows the higher-level models in the utils/inc_net.py file to:
  1. Use the convnet for feature extraction
  2. Add their own classification layers
  3. Implement specialized mechanisms like distillation, attention, or feature transformation

  When the full model is executed, the convnet output with feature maps and features is passed to the classifier layers, which add the logits to create the complete output that includes both features and
  classification results.