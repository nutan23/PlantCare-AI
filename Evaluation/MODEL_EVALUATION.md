# PlantCare Model Evaluation

## Test Dataset

The PlantCare disease detection model was evaluated using:

-   15 dataset images
-   15 external real-world images
-   5 uncertain / out-of-distribution (OOD) images

**Total test images: 35**

## Evaluation Results

  Test Type                       Accuracy
  ----------------------------- ----------
  Dataset Images                      100%
  External Raw Classification       66.67%
  Uncertain / OOD Rejection            80%
  Overall System Result             68.57%

## External Fine-Tuning Experiment

The existing real-world model and the PlantWild fine-tuned model were
compared using 10 external images.

  Model                          Accuracy
  ---------------------------- ----------
  Existing Real-World Model           60%
  PlantWild Fine-Tuned Model          60%

Although PlantWild fine-tuning improved predictions for some individual
images, the overall external accuracy remained the same. Therefore, the
existing `realworld_best_model.keras` model was retained for deployment.

## Observations

-   The model performs strongly on dataset-style images.
-   Real-world images are more challenging because of differences in
    lighting, background, camera angle, image quality, leaf orientation,
    and disease severity.
-   A confidence threshold is used to prevent the system from presenting
    low-confidence predictions as definite disease diagnoses.
-   OOD/uncertain testing was included to evaluate how the system
    behaves when an image cannot be classified confidently.

## Deployed Model

`model/realworld_best_model.keras`

Class labels:

`model/class_names.json`

## Disclaimer

PlantCare is an educational and decision-support project. Disease
predictions and treatment guidance should not replace professional
agricultural advice or laboratory diagnosis.
