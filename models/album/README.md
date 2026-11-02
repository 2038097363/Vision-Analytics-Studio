# Local album classifier

The `.pth` weight file is intentionally ignored. Development uses the official torchvision URL recorded in `manifest.json`; release builds must download it on user request, verify the exact byte size and SHA-256, and store it under the application data directory.

The current mapping only emits eight coarse, non-sensitive categories. It does not run OCR, face recognition, identity clustering, GPS classification, or open-ended attribute inference. The model's distribution and dataset terms still require release review, so the weight is not bundled in a store package.
