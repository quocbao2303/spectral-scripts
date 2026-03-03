#╔════════════════════════════════════════════════════════════════════╗
#║             SPECTRAL SCRIPTS PIPELINE - MAKEFILE                   ║
#║  Text → Images → OCR → Confusion Matrices → Spectral Analysis      ║
#║                                                                    ║
#║  A complete pipeline for analyzing character confusion patterns    ║
#║  across different writing systems using spectral methods.          ║
#║                                                                    ║
#╚════════════════════════════════════════════════════════════════════╝

#════════════════════════════════════════════════════════════════════
# METADATA & VERSION
#════════════════════════════════════════════════════════════════════

PIPELINE_VERSION := 2.1
PIPELINE_DATE := 2025-12-03
MAKEFILE_DIR := $(dir $(abspath $(firstword $(MAKEFILE_LIST))))

#════════════════════════════════════════════════════════════════════
# PHONY TARGETS
#════════════════════════════════════════════════════════════════════

.PHONY: help version \
        install setup clean distclean \
        check-env check-python check-dependencies preflight \
        text-help text-setup text-generate text-validate text-clean \
        ocr-help ocr-setup ocr-validate ocr-process ocr-clean \
        analysis-help analyze quick-analyze validate visualize report \
        pipeline trocr-pipeline tesseract-pipeline easyocr-pipeline \
        paddle-pipeline surya-pipeline glm-pipeline \
        status summary logs debug debug-trocr debug-ocr debug-all \
        install-trocr install-tesseract install-easyocr install-paddle \
        install-surya install-glm install-all-ocr

.SUFFIXES:

#════════════════════════════════════════════════════════════════════
# CONFIGURATION VARIABLES
#════════════════════════════════════════════════════════════════════

# Python
PYTHON := python
PIP := pip
PYTEST := pytest

# Project structure
PROJECT_ROOT := $(MAKEFILE_DIR)
SRC_DIR := $(PROJECT_ROOT)src/spectral_scripts
SCRIPTS_DIR := $(PROJECT_ROOT)scripts
DATA_DIR := $(PROJECT_ROOT)data
OUTPUT_DIR := $(PROJECT_ROOT)outputs
LOGS_DIR := $(OUTPUT_DIR)/_logs

# Stage 1: Text-to-Image
TEXT_INPUT_DIR := $(PROJECT_ROOT)data/raw/texts
TEXT_IMAGES_DIR := $(PROJECT_ROOT)data/raw/images
TEXT_GT_DIR := $(PROJECT_ROOT)data/raw/ground_truth
DISCOVERED_SCRIPTS := $(shell if [ -d "$(TEXT_INPUT_DIR)" ]; then ls -d $(TEXT_INPUT_DIR)/*/ 2>/dev/null | xargs -I {} basename {} | sort; fi)
SCRIPTS ?= $(if $(DISCOVERED_SCRIPTS),$(DISCOVERED_SCRIPTS),latin greek cyrillic arabic)

# OCR Configuration
OCR_ENGINE ?= trocr
OCR_MODEL_TYPE ?= printed
OCR_DEVICE ?= cpu
OCR_BATCH_SIZE ?= 8
OCR_CONFIG ?= ocr_config.yaml

# Stage 2: OCR Pipeline
OCR_INPUT_DIR := $(TEXT_IMAGES_DIR)
OCR_GT_DIR := $(TEXT_GT_DIR)
OCR_OUTPUT_DIR := $(DATA_DIR)/confusion_matrices/$(OCR_ENGINE)

# Stage 3: Analysis
ANALYSIS_INPUT_DIR := $(OCR_OUTPUT_DIR)
ANALYSIS_OUTPUT_DIR := $(OUTPUT_DIR)/$(OCR_ENGINE)

# Color codes
RESET := \033[0m
BOLD := \033[1m
DIM := \033[2m

# Colors
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
MAGENTA := \033[0;35m
CYAN := \033[0;36m
WHITE := \033[0;37m

# Combined styles
SUCCESS := $(GREEN)$(BOLD)
ERROR := $(RED)$(BOLD)
WARNING := $(YELLOW)$(BOLD)
INFO := $(CYAN)
SECTION := $(MAGENTA)$(BOLD)
HIGHLIGHT := $(YELLOW)

#════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
#════════════════════════════════════════════════════════════════════

# Echo functions
define echo_header
printf "\n$(SECTION)╔════════════════════════════════════════════════════════════╗$(RESET)\n"
endef

define echo_title
printf "$(SECTION)║ %-58s ║$(RESET)\n" "$(1)"
endef

define echo_footer
printf "$(SECTION)╚════════════════════════════════════════════════════════════╝$(RESET)\n\n"
endef

define echo_step
printf "$(INFO)▶ $(BOLD)$(1)$(RESET)\n"
endef

define echo_success
printf "  $(SUCCESS)✓ $(1)$(RESET)\n"
endef

define echo_error
printf "  $(ERROR)✗ $(1)$(RESET)\n"
endef

define echo_warning
printf "  $(WARNING)⚠ $(1)$(RESET)\n"
endef

define echo_info
printf "  $(INFO)ℹ $(1)$(RESET)\n"
endef

# Create logs directory
$(LOGS_DIR):
	@mkdir -p $(LOGS_DIR)

#════════════════════════════════════════════════════════════════════
# MAIN HELP TARGET
#════════════════════════════════════════════════════════════════════

help:
	@$(call echo_header)
	@$(call echo_title,"SPECTRAL SCRIPTS PIPELINE v$(PIPELINE_VERSION)")
	@$(call echo_title,"Complete OCR → Confusion Matrix → Analysis")
	@$(call echo_footer)
	@printf "$(BOLD)SUPPORTED LANGUAGES$(RESET)\n\n"
	@printf "  Latin, Greek, Cyrillic, Arabic\n"
	@printf "  (Extensible - add more via config.py)\n\n"
	@printf "$(BOLD)QUICK START$(RESET)\n\n"
	@printf "  $(HIGHLIGHT)Complete pipeline (all stages):$(RESET)\n"
	@printf "    $(CYAN)make trocr-pipeline$(RESET)                    All languages with TrOCR\n"
	@printf "    $(CYAN)make paddle-pipeline$(RESET)                   All languages with PaddleOCR (recommended for Arabic)\n"
	@printf "    $(CYAN)make glm-pipeline$(RESET)                      All languages with GLM-OCR (VLM-based)\n"
	@printf "    $(CYAN)make surya-pipeline$(RESET)                    All languages with Surya (high accuracy)\n"
	@printf "    $(CYAN)make paddle-pipeline SCRIPTS=arabic$(RESET)    Arabic only with PaddleOCR\n"
	@printf "    $(CYAN)make glm-pipeline SCRIPTS=latin$(RESET)        Latin only with GLM-OCR\n"
	@printf "    $(CYAN)make tesseract-pipeline SCRIPTS='latin greek'$(RESET) Latin + Greek\n\n"
	@printf "  $(HIGHLIGHT)Individual stages:$(RESET)\n"
	@printf "    $(CYAN)make text-generate$(RESET)     - Generate images from text\n"
	@printf "    $(CYAN)make ocr-process$(RESET)      - Run OCR on images (default: trocr)\n"
	@printf "    $(CYAN)make analyze$(RESET)          - Run spectral analysis\n\n"
	@printf "$(BOLD)DETAILED HELP$(RESET)\n\n"
	@printf "  $(CYAN)make text-help$(RESET)         - Text-to-image pipeline help\n"
	@printf "  $(CYAN)make ocr-help$(RESET)          - OCR pipeline help\n"
	@printf "  $(CYAN)make analysis-help$(RESET)     - Analysis pipeline help\n"
	@printf "  $(CYAN)make version$(RESET)           - Show version info\n"
	@printf "  $(CYAN)make status$(RESET)            - Check pipeline status\n\n"
	@printf "  $(CYAN)make setup$(RESET)             - Create directory structure\n"
	@printf "  $(CYAN)make install$(RESET)           - Install core dependencies\n"
	@printf "  $(CYAN)make install-glm$(RESET)       - Install GLM-OCR (requires GitHub source transformers)\n"
	@printf "  $(CYAN)make install-all-ocr$(RESET)   - Install all support OCR engines\n"
	@printf "  $(CYAN)make check-env$(RESET)         - Check environment\n"
	@printf "  $(CYAN)make clean$(RESET)             - Clean outputs only\n"
	@printf "  $(CYAN)make distclean$(RESET)         - Full cleanup (keep data)\n\n"

version:
	@$(call echo_header)
	@$(call echo_title,"SPECTRAL SCRIPTS PIPELINE INFO")
	@$(call echo_footer)
	@printf "Version:      $(HIGHLIGHT)$(PIPELINE_VERSION)$(RESET)\n"
	@printf "Date:         $(HIGHLIGHT)$(PIPELINE_DATE)$(RESET)\n"
	@printf "Python:       $(HIGHLIGHT)$(PYTHON)$(RESET)\n"
	@printf "Location:     $(HIGHLIGHT)$(PROJECT_ROOT)$(RESET)\n"
	@printf "Languages:    $(HIGHLIGHT)Latin, Greek, Cyrillic, Arabic$(RESET)\n"
	@printf "Detected:     $(HIGHLIGHT)$(SCRIPTS)$(RESET)\n\n"

#════════════════════════════════════════════════════════════════════
# ENVIRONMENT CHECKS
#════════════════════════════════════════════════════════════════════

check-python:
	@$(call echo_step,"Checking Python installation")
	@command -v $(PYTHON) >/dev/null 2>&1 || \
		($(call echo_error,"Python 3 not found") && exit 1)
	@$(call echo_success,"Python found: $$($(PYTHON) --version)")

check-dependencies: | $(LOGS_DIR)
	@$(call echo_step,"Checking Python dependencies")
	@$(PYTHON) -c "import numpy, pandas, scipy, matplotlib, seaborn; print('ok')" 2>/dev/null && \
		$(call echo_success,"All dependencies installed") || \
		$(call echo_warning,"Some dependencies missing - run 'make install'")

check-env: check-python check-dependencies
	@$(call echo_step,"Checking directory structure")
	@test -d $(DATA_DIR) && $(call echo_success,"Data directory exists") || \
		($(call echo_warning,"Data directory missing") && mkdir -p $(DATA_DIR))
	@test -d $(OUTPUT_DIR) && $(call echo_success,"Output directory exists") || \
		($(call echo_warning,"Output directory missing") && mkdir -p $(OUTPUT_DIR))

preflight: check-env
	@$(call echo_step,"Running preflight checks")
	@$(PYTHON) -c "from pathlib import Path; import sys" 2>/dev/null && \
		$(call echo_success,"Preflight checks passed") || exit 1

#════════════════════════════════════════════════════════════════════
# INSTALLATION & SETUP
#════════════════════════════════════════════════════════════════════

setup:
	@$(call echo_header)
	@$(call echo_title,"CREATING DIRECTORY STRUCTURE")
	@$(call echo_footer)
	@$(call echo_step,"Creating directories for scripts: $(SCRIPTS)")
	@mkdir -p $(TEXT_INPUT_DIR)
	@mkdir -p $(TEXT_IMAGES_DIR)
	@mkdir -p $(TEXT_GT_DIR)
	@mkdir -p $(LOGS_DIR)
	@for script in $(SCRIPTS); do \
		mkdir -p $(TEXT_INPUT_DIR)/$$script; \
		mkdir -p $(TEXT_IMAGES_DIR)/$$script; \
	done
	@mkdir -p $(OCR_OUTPUT_DIR)
	@mkdir -p $(ANALYSIS_OUTPUT_DIR)/figures $(ANALYSIS_OUTPUT_DIR)/validation
	@$(call echo_success,"Directory structure created")
	@printf "\n$(BOLD)Place your text files in:$(RESET)\n"
	@for script in $(SCRIPTS); do \
		printf "  $(HIGHLIGHT)$(TEXT_INPUT_DIR)/$$script/$(RESET)\n"; \
	done
	@printf "\n"

install: check-python
	@$(call echo_header)
	@$(call echo_title,"INSTALLING PYTHON DEPENDENCIES")
	@$(call echo_footer)
	@$(call echo_step,"Installing core dependencies")
	@$(PIP) install -e . 2>&1 | tail -5
	@$(call echo_success,"Dependencies installed")

install-trocr:
	@$(call echo_header)
	@$(call echo_title,"INSTALLING TROCR DEPENDENCIES")
	@$(call echo_footer)
	@$(call echo_step,"Installing TrOCR packages")
	@$(PIP) install torch transformers pillow opencv-python pyyaml "numpy<2" 2>&1 | tail -5
	@$(call echo_success,"TrOCR dependencies installed")
	@printf "\n$(HIGHLIGHT)Usage:$(RESET)\n"
	@printf "  $(CYAN)make ocr-process OCR_ENGINE=trocr$(RESET)\n"
	@printf "  $(CYAN)make trocr-pipeline$(RESET)\n\n"

install-tesseract:
	@$(call echo_header)
	@$(call echo_title,"INSTALLING TESSERACT DEPENDENCIES")
	@$(call echo_footer)
	@$(call echo_step,"Installing Tesseract packages")
	@$(PIP) install pytesseract pillow opencv-python pyyaml "numpy<2" 2>&1 | tail -5
	@$(call echo_warning,"Tesseract binary still needs to be installed on your system")
	@printf "  macOS:  $(CYAN)brew install tesseract$(RESET)\n"
	@printf "  Linux:  $(CYAN)sudo apt install tesseract-ocr$(RESET)\n"
	@printf "  Win:    https://github.com/UB-Mannheim/tesseract/wiki\n\n"

install-easyocr:
	@$(call echo_header)
	@$(call echo_title,"INSTALLING EASYOCR DEPENDENCIES")
	@$(call echo_footer)
	@$(call echo_step,"Installing EasyOCR packages")
	@$(PIP) install easyocr pillow opencv-python pyyaml "numpy<2" 2>&1 | tail -5
	@$(call echo_success,"EasyOCR dependencies installed")
	@printf "\n$(HIGHLIGHT)Usage:$(RESET)\n"
	@printf "  $(CYAN)make ocr-process OCR_ENGINE=easyocr SCRIPTS='latin cyrillic'$(RESET)\n\n"

install-paddle:
	@$(call echo_header)
	@$(call echo_title,"INSTALLING PADDLEOCR DEPENDENCIES")
	@$(call echo_footer)
	@$(call echo_step,"Installing PaddleOCR packages")
	@$(PIP) install paddlepaddle paddleocr pillow opencv-python pyyaml "numpy<2" 2>&1 | tail -5
	@$(call echo_success,"PaddleOCR dependencies installed")
	@printf "\n$(HIGHLIGHT)Usage:$(RESET)\n"
	@printf "  $(CYAN)make ocr-process OCR_ENGINE=paddle$(RESET)\n"
	@printf "  $(CYAN)make paddle-pipeline$(RESET)\n"
	@printf "  $(CYAN)make paddle-pipeline SCRIPTS=arabic$(RESET)\n\n"

install-surya:
	@$(call echo_header)
	@$(call echo_title,"INSTALLING SURYA OCR DEPENDENCIES")
	@$(call echo_footer)
	@$(call echo_step,"Installing Surya OCR packages")
	@$(PIP) install surya-ocr torch transformers pillow opencv-python pyyaml "numpy<2" 2>&1 | tail -5
	@$(call echo_success,"Surya OCR dependencies installed")
	@printf "\n$(HIGHLIGHT)Usage:$(RESET)\n"
	@printf "  $(CYAN)make ocr-process OCR_ENGINE=surya$(RESET)\n"
	@printf "  $(CYAN)make surya-pipeline$(RESET)\n\n"

install-glm:
	@$(call echo_header)
	@$(call echo_title,"INSTALLING GLM-OCR DEPENDENCIES")
	@$(call echo_footer)
	@$(call echo_step,"Installing GLM-OCR packages - source transformers required")
	@$(PIP) install "git+https://github.com/huggingface/transformers.git" torchvision accelerate pillow opencv-python pyyaml "numpy<2" 2>&1 | tail -5
	@$(call echo_success,"GLM-OCR dependencies installed")
	@printf "\n$(HIGHLIGHT)Usage:$(RESET)\n"
	@printf "  $(CYAN)make ocr-process OCR_ENGINE=glm$(RESET)\n"
	@printf "  $(CYAN)make glm-pipeline$(RESET)\n\n"

install-all-ocr: install-trocr install-tesseract install-easyocr install-paddle install-surya install-glm
	@$(call echo_header)
	@$(call echo_title,"ALL OCR ENGINES INSTALLED")
	@$(call echo_footer)
	@$(SUCCESS)✓ Available engines: TrOCR, Tesseract, EasyOCR, PaddleOCR, Surya, GLM-OCR$(RESET)\n\n"
	@printf "$(HIGHLIGHT)Quick commands:$(RESET)\n"
	@printf "  $(CYAN)make trocr-pipeline$(RESET)\n"
	@printf "  $(CYAN)make paddle-pipeline$(RESET)\n"
	@printf "  $(CYAN)make glm-pipeline$(RESET)\n"
	@printf "  $(CYAN)make surya-pipeline$(RESET)\n"
	@printf "  $(CYAN)make tesseract-pipeline$(RESET)\n"
	@printf "  $(CYAN)make easyocr-pipeline$(RESET)\n\n"

#════════════════════════════════════════════════════════════════════
# STAGE 1: TEXT-TO-IMAGE PIPELINE
#════════════════════════════════════════════════════════════════════

text-help:
	@$(call echo_header)
	@$(call echo_title,"TEXT-TO-IMAGE PIPELINE")
	@$(call echo_footer)
	@printf "$(BOLD)Converts text files to rendered images with ground truth labels$(RESET)\n\n"
	@printf "$(HIGHLIGHT)Basic usage:$(RESET)\n"
	@printf "  $(CYAN)make text-generate$(RESET)                      Auto-discover all scripts\n"
	@printf "  $(CYAN)make text-generate SCRIPTS=arabic$(RESET)       Only Arabic script\n"
	@printf "  $(CYAN)make text-generate SCRIPTS='latin arabic'$(RESET) Multiple scripts\n"
	@printf "  $(CYAN)make text-validate$(RESET)                      Validate text files before processing\n"
	@printf "  $(CYAN)make text-clean$(RESET)                         Remove generated images\n\n"
	@printf "$(HIGHLIGHT)Input format:$(RESET)\n"
	@printf "  Create folders in: $(TEXT_INPUT_DIR)/\n"
	@printf "  Each folder name = script (latin, greek, cyrillic, arabic, etc.)\n"
	@printf "  Place .txt files inside (one sentence per line)\n\n"
	@printf "$(HIGHLIGHT)Currently configured scripts:$(RESET)\n"
	@for script in $(SCRIPTS); do printf "  • $$script\n"; done
	@printf "\n$(HIGHLIGHT)Output:$(RESET)\n"
	@printf "  Images:       $(TEXT_IMAGES_DIR)/{script}/\n"
	@printf "  Ground truth: $(TEXT_GT_DIR)/{script}_labels.txt\n\n"

text-setup:
	@$(call echo_step,"Setting up text-to-image directories")
	@mkdir -p $(TEXT_INPUT_DIR)
	@for script in $(SCRIPTS); do mkdir -p $(TEXT_INPUT_DIR)/$$script; done
	@$(call echo_success,"Directories ready for: $(SCRIPTS)")

text-validate: text-setup
	@$(call echo_header)
	@$(call echo_title,"VALIDATING TEXT INPUT FILES")
	@$(call echo_footer)
	@$(call echo_step,"Checking text files for: $(SCRIPTS)")
	@for script in $(SCRIPTS); do \
		count=$$(ls -1 $(TEXT_INPUT_DIR)/$$script/*.txt 2>/dev/null | wc -l); \
		if [ $$count -eq 0 ]; then \
			printf "  $(WARNING)⚠ No .txt files in $(TEXT_INPUT_DIR)/$$script/$(RESET)\n"; \
		else \
			printf "  $(SUCCESS)✓ $$script: $$count file(s)$(RESET)\n"; \
		fi; \
	done
	@printf "\n$(HIGHLIGHT)Next step:$(RESET)\n"
	@printf "  $(CYAN)make text-generate$(RESET)\n\n"

text-generate: text-validate preflight | $(LOGS_DIR)
	@$(call echo_header)
	@$(call echo_title,"STAGE 1: TEXT-TO-IMAGE PIPELINE")
	@$(call echo_footer)
	@$(call echo_step,"Generating images from text for: $(SCRIPTS)")
	@$(PYTHON) $(SCRIPTS_DIR)/run_text_to_image.py \
		--input-dir $(TEXT_INPUT_DIR) \
		--output-dir $(DATA_DIR)/raw \
		--scripts $(SCRIPTS) \
		--verbose 2>&1 | tee $(LOGS_DIR)/text-to-image.log
	@$(call echo_success,"Text-to-image stage complete")
	@printf "\n$(HIGHLIGHT)Generated:$(RESET)\n"
	@printf "  Images:       $(TEXT_IMAGES_DIR)/\n"
	@printf "  Ground truth: $(TEXT_GT_DIR)/\n"
	@printf "  Logs:         $(LOGS_DIR)/text-to-image.log\n\n"

text-clean:
	@$(call echo_header)
	@$(call echo_title,"CLEANING TEXT-TO-IMAGE OUTPUTS")
	@$(call echo_footer)
	@$(call echo_step,"Removing generated images and ground truth")
	@rm -rf $(TEXT_IMAGES_DIR)/*/*.png
	@rm -rf $(TEXT_GT_DIR)/*.txt
	@rm -f $(DATA_DIR)/raw/dataset_summary.json
	@$(call echo_success,"Text-to-image outputs cleaned\n")

#════════════════════════════════════════════════════════════════════
# STAGE 2: OCR PIPELINE
#════════════════════════════════════════════════════════════════════

ocr-help:
	@$(call echo_header)
	@$(call echo_title,"OCR PIPELINE")
	@$(call echo_footer)
	@printf "$(BOLD)Runs OCR on images to generate confusion matrices$(RESET)\n\n"
	@printf "$(HIGHLIGHT)Basic usage:$(RESET)\n"
	@printf "  $(CYAN)make ocr-process$(RESET)                              Run OCR with default engine (trocr)\n"
	@printf "  $(CYAN)make ocr-process OCR_ENGINE=paddle$(RESET)            Use PaddleOCR (recommended for Arabic)\n"
	@printf "  $(CYAN)make ocr-process OCR_ENGINE=surya$(RESET)             Use Surya (highest accuracy)\n"
	@printf "  $(CYAN)make ocr-process OCR_ENGINE=tesseract$(RESET)         Use Tesseract\n"
	@printf "  $(CYAN)make ocr-process OCR_ENGINE=easyocr$(RESET)           Use EasyOCR (Latin+Cyrillic only)\n"
	@printf "  $(CYAN)make ocr-validate$(RESET)                             Validate inputs (dry-run)\n"
	@printf "  $(CYAN)make ocr-clean$(RESET)                                Remove confusion matrices\n\n"
	@printf "$(HIGHLIGHT)Language support by engine:$(RESET)\n"
	@printf "  TrOCR:     Latin, Greek, Cyrillic, Arabic\n"
	@printf "  PaddleOCR: Latin, Greek, Cyrillic, Arabic (★ Best for Arabic)\n"
	@printf "  Surya:     Latin, Greek, Cyrillic, Arabic (★ Highest accuracy)\n"
	@printf "  Tesseract: Latin, Greek, Cyrillic, Arabic\n"
	@printf "  EasyOCR:   Latin, Cyrillic (not Greek or Arabic)\n\n"
	@printf "$(HIGHLIGHT)Device options (TrOCR, Surya, PaddleOCR):$(RESET)\n"
	@printf "  $(CYAN)OCR_DEVICE=cpu$(RESET)                               CPU only (default)\n"
	@printf "  $(CYAN)OCR_DEVICE=cuda$(RESET)                              NVIDIA GPU\n"
	@printf "  $(CYAN)OCR_DEVICE=mps$(RESET)                               Apple Silicon\n\n"
	@printf "$(HIGHLIGHT)TrOCR model type:$(RESET)\n"
	@printf "  $(CYAN)OCR_MODEL_TYPE=printed$(RESET)                       Printed text (default)\n"
	@printf "  $(CYAN)OCR_MODEL_TYPE=handwritten$(RESET)                   Handwritten text\n\n"
	@printf "$(HIGHLIGHT)Examples:$(RESET)\n"
	@printf "  $(CYAN)make ocr-process OCR_ENGINE=paddle SCRIPTS=arabic$(RESET)\n"
	@printf "  $(CYAN)make ocr-process OCR_ENGINE=surya OCR_DEVICE=cuda$(RESET)\n"
	@printf "  $(CYAN)make ocr-process OCR_ENGINE=trocr OCR_MODEL_TYPE=handwritten$(RESET)\n"
	@printf "  $(CYAN)make ocr-process OCR_ENGINE=easyocr SCRIPTS='latin cyrillic'$(RESET)\n\n"
	@printf "$(HIGHLIGHT)Input:$(RESET)\n"
	@printf "  Images:       $(OCR_INPUT_DIR)/{script}/\n"
	@printf "  Ground truth: $(OCR_GT_DIR)/{script}_labels.txt\n\n"
	@printf "$(HIGHLIGHT)Output:$(RESET)\n"
	@printf "  Matrices: $(OCR_OUTPUT_DIR)/{script}.npz\n"
	@printf "  Report:   $(OCR_OUTPUT_DIR)/pipeline_report.md\n"
	@printf "  Results:  $(OCR_OUTPUT_DIR)/pipeline_results.json\n\n"

ocr-setup:
	@$(call echo_step,"Setting up OCR directories")
	@mkdir -p $(OCR_OUTPUT_DIR)
	@$(call echo_success,"OCR directories ready")

ocr-validate: ocr-setup preflight | $(LOGS_DIR)
	@$(call echo_header)
	@$(call echo_title,"OCR PIPELINE - DRY RUN (VALIDATION)")
	@$(call echo_footer)
	@$(call echo_step,"Validating OCR inputs and configuration")
	@printf "\n$(HIGHLIGHT)Configuration:$(RESET)\n"
	@printf "  Engine:        $(OCR_ENGINE)\n"
	@printf "  Device:        $(OCR_DEVICE)\n"
	@printf "  Batch size:    $(OCR_BATCH_SIZE)\n"
	@printf "  Scripts:       $(SCRIPTS)\n"
	@if [ "$(OCR_ENGINE)" = "trocr" ]; then printf "  Model type:    $(OCR_MODEL_TYPE)\n"; fi
	@printf "  Input:         $(OCR_INPUT_DIR)\n"
	@printf "  Output:        $(OCR_OUTPUT_DIR)\n\n"
	@$(PYTHON) $(SCRIPTS_DIR)/run_ocr_pipeline.py \
		--engine $(OCR_ENGINE) \
		--device $(OCR_DEVICE) \
		--batch-size $(OCR_BATCH_SIZE) \
		--scripts $(SCRIPTS) \
		--input-dir $(OCR_INPUT_DIR) \
		--ground-truth-dir $(OCR_GT_DIR) \
		--output-dir $(OCR_OUTPUT_DIR) \
		$(if $(filter trocr,$(OCR_ENGINE)),--trocr-model-type $(OCR_MODEL_TYPE)) \
		--dry-run \
		--verbose 2>&1 | tee $(LOGS_DIR)/ocr-validate.log
	@$(call echo_success,"Validation complete")
	@printf "\n$(HIGHLIGHT)Next step:$(RESET)\n"
	@printf "  $(CYAN)make ocr-process$(RESET)\n\n"

ocr-process: ocr-setup preflight | $(LOGS_DIR)
	@$(call echo_header)
	@$(call echo_title,"STAGE 2: OCR PIPELINE - $(OCR_ENGINE)")
	@$(call echo_footer)
	@$(call echo_step,"Running OCR engine: $(OCR_ENGINE)")
	@printf "\n$(HIGHLIGHT)Configuration:$(RESET)\n"
	@printf "  Engine:        $(OCR_ENGINE)\n"
	@printf "  Device:        $(OCR_DEVICE)\n"
	@printf "  Batch size:    $(OCR_BATCH_SIZE)\n"
	@printf "  Scripts:       $(SCRIPTS)\n"
	@if [ "$(OCR_ENGINE)" = "trocr" ]; then printf "  Model type:    $(OCR_MODEL_TYPE)\n"; fi
	@printf "  Input:         $(OCR_INPUT_DIR)\n"
	@printf "  Output:        $(OCR_OUTPUT_DIR)\n\n"
	@$(PYTHON) $(SCRIPTS_DIR)/run_ocr_pipeline.py \
		--engine $(OCR_ENGINE) \
		--device $(OCR_DEVICE) \
		--batch-size $(OCR_BATCH_SIZE) \
		--scripts $(SCRIPTS) \
		--input-dir $(OCR_INPUT_DIR) \
		--ground-truth-dir $(OCR_GT_DIR) \
		--output-dir $(OCR_OUTPUT_DIR) \
		$(if $(filter trocr,$(OCR_ENGINE)),--trocr-model-type $(OCR_MODEL_TYPE)) \
		--verbose 2>&1 | tee $(LOGS_DIR)/ocr-process-$(OCR_ENGINE).log
	@$(call echo_success,"OCR processing complete")
	@printf "\n$(HIGHLIGHT)Verifying matrices...$(RESET)\n"
	@$(PYTHON) -c "from pathlib import Path; from spectral_scripts.core.confusion_matrix import ConfusionMatrix; ocr_dir = Path('$(OCR_OUTPUT_DIR)'); matrices = [ConfusionMatrix.from_npz(npz_file) for npz_file in sorted(ocr_dir.glob('*.npz'))]; [print(f'  {cm.script}: {cm.size}×{cm.size}, {cm.total_observations:,} obs, {cm.accuracy:.1%} acc') for cm in matrices]; print(f'\nTotal matrices: {len(matrices)}')" 2>/dev/null || $(call echo_warning,"Could not verify matrices")
	@printf "\n$(HIGHLIGHT)Outputs:$(RESET)\n"
	@printf "  Matrices: $(OCR_OUTPUT_DIR)/*.npz\n"
	@printf "  Report:   $(OCR_OUTPUT_DIR)/pipeline_report.md\n"
	@printf "  Results:  $(OCR_OUTPUT_DIR)/pipeline_results.json\n"
	@printf "  Logs:     $(LOGS_DIR)/ocr-process-$(OCR_ENGINE).log\n\n"

ocr-clean:
	@$(call echo_header)
	@$(call echo_title,"CLEANING OCR OUTPUTS")
	@$(call echo_footer)
	@$(call echo_step,"Removing confusion matrices and OCR outputs")
	@rm -f $(OCR_OUTPUT_DIR)/*.npz
	@rm -f $(OCR_OUTPUT_DIR)/*.json
	@rm -f $(OCR_OUTPUT_DIR)/*.md
	@$(call echo_success,"OCR outputs cleaned\n")

#════════════════════════════════════════════════════════════════════
# STAGE 3: SPECTRAL ANALYSIS
#════════════════════════════════════════════════════════════════════

analysis-help:
	@$(call echo_header)
	@$(call echo_title,"SPECTRAL ANALYSIS PIPELINE")
	@$(call echo_footer)
	@printf "$(BOLD)Analyzes confusion matrices using spectral methods$(RESET)\n\n"
	@printf "$(HIGHLIGHT)Basic usage:$(RESET)\n"
	@printf "  $(CYAN)make analyze$(RESET)          Run full analysis (all methods)\n"
	@printf "  $(CYAN)make quick-analyze$(RESET)    Fast analysis (spectral only)\n"
	@printf "  $(CYAN)make validate$(RESET)         Run validation tests\n"
	@printf "  $(CYAN)make report$(RESET)           Generate markdown report\n"
	@printf "  $(CYAN)make visualize$(RESET)        Show visualizations\n\n"
	@printf "$(HIGHLIGHT)Input:$(RESET)\n"
	@printf "  Matrices: $(ANALYSIS_INPUT_DIR)/{script}.npz\n\n"
	@printf "$(HIGHLIGHT)Output:$(RESET)\n"
	@printf "  Results:      $(ANALYSIS_OUTPUT_DIR)/results.json\n"
	@printf "  Figures:      $(ANALYSIS_OUTPUT_DIR)/figures/\n"
	@printf "  Report:       $(ANALYSIS_OUTPUT_DIR)/report.md\n"
	@printf "  Validation:   $(ANALYSIS_OUTPUT_DIR)/validation/\n\n"

analyze: preflight | $(LOGS_DIR)
	@$(call echo_header)
	@$(call echo_title,"STAGE 3: SPECTRAL ANALYSIS PIPELINE")
	@$(call echo_footer)
	@$(call echo_step,"Checking confusion matrices for: $(SCRIPTS)")
	@for script in $(SCRIPTS); do \
		if [ ! -f $(ANALYSIS_INPUT_DIR)/$$script.npz ]; then \
			$(call echo_error,"Missing: $(ANALYSIS_INPUT_DIR)/$$script.npz"); \
			exit 1; \
		fi; \
	done
	@$(call echo_success,"All matrices present for: $(SCRIPTS)")
	@$(call echo_step,"Running spectral analysis")
	@$(PYTHON) $(SCRIPTS_DIR)/run_analysis.py \
		--input-dir $(ANALYSIS_INPUT_DIR) \
		--output-dir $(ANALYSIS_OUTPUT_DIR) \
		--method spectral 2>&1 | tee $(LOGS_DIR)/analysis-spectral.log
	@$(call echo_success,"Spectral analysis complete")
	@$(call echo_step,"Running multi-spectrum analysis")
	@$(PYTHON) $(SCRIPTS_DIR)/run_analysis.py \
		--input-dir $(ANALYSIS_INPUT_DIR) \
		--output-dir $(ANALYSIS_OUTPUT_DIR)/multi_spectrum \
		--method multi_spectrum 2>&1 | tee $(LOGS_DIR)/analysis-multispectrum.log
	@$(call echo_success,"Multi-spectrum analysis complete")
	@$(call echo_step,"Running Frobenius baseline")
	@$(PYTHON) $(SCRIPTS_DIR)/run_analysis.py \
		--input-dir $(ANALYSIS_INPUT_DIR) \
		--output-dir $(ANALYSIS_OUTPUT_DIR)/frobenius \
		--method frobenius 2>&1 | tee $(LOGS_DIR)/analysis-frobenius.log
	@$(call echo_success,"Frobenius analysis complete")
	@printf "\n$(HIGHLIGHT)Outputs:$(RESET)\n"
	@printf "  Results:    $(ANALYSIS_OUTPUT_DIR)/results.json\n"
	@printf "  Figures:    $(ANALYSIS_OUTPUT_DIR)/figures/\n"
	@printf "  Logs:       $(LOGS_DIR)/analysis-*.log\n\n"

quick-analyze: preflight | $(LOGS_DIR)
	@$(call echo_header)
	@$(call echo_title,"QUICK ANALYSIS - SPECTRAL METHOD ONLY")
	@$(call echo_footer)
	@$(call echo_step,"Running spectral analysis (fast mode)")
	@$(PYTHON) $(SCRIPTS_DIR)/run_analysis.py \
		--input-dir $(ANALYSIS_INPUT_DIR) \
		--output-dir $(ANALYSIS_OUTPUT_DIR) \
		--method spectral 2>&1 | tee $(LOGS_DIR)/analysis-quick.log
	@$(call echo_success,"Quick analysis complete\n")

validate: preflight | $(LOGS_DIR)
	@$(call echo_header)
	@$(call echo_title,"VALIDATION TESTS")
	@$(call echo_footer)
	@$(call echo_step,"Running synthetic validation")
	@$(PYTHON) $(SCRIPTS_DIR)/run_synthetic_validation.py \
		--n-matrices 5 \
		--output-dir $(ANALYSIS_OUTPUT_DIR)/validation 2>&1 | tee $(LOGS_DIR)/validation.log
	@$(call echo_success,"Validation tests complete\n")

report:
	@$(call echo_header)
	@$(call echo_title,"GENERATING MARKDOWN REPORT")
	@$(call echo_footer)
	@$(call echo_step,"Creating report")
	@$(PYTHON) $(SCRIPTS_DIR)/generate_report.py \
		--results-dir $(ANALYSIS_OUTPUT_DIR) \
		--output $(ANALYSIS_OUTPUT_DIR)/report.md 2>&1 | tee $(LOGS_DIR)/report.log
	@$(call echo_success,"Report generated: $(ANALYSIS_OUTPUT_DIR)/report.md")
	@if [ -f $(ANALYSIS_OUTPUT_DIR)/report.md ]; then \
		printf "\n$(HIGHLIGHT)Report preview:$(RESET)\n"; \
		head -30 $(ANALYSIS_OUTPUT_DIR)/report.md | sed 's/^/  /'; \
		printf "  ...\n\n"; \
	fi

visualize:
	@$(call echo_header)
	@$(call echo_title,"VIEWING VISUALIZATIONS")
	@$(call echo_footer)
	@if [ -d $(ANALYSIS_OUTPUT_DIR)/figures ] && [ -n "$$(ls -A $(ANALYSIS_OUTPUT_DIR)/figures/ 2>/dev/null)" ]; then \
		$(call echo_success,"Generated figures:"); \
		ls -lh $(ANALYSIS_OUTPUT_DIR)/figures/ | tail -n +2 | awk '{print "  " $$9 " (" $$5 ")"}'; \
	else \
		$(call echo_warning,"No figures generated yet - run 'make analyze' first"); \
	fi
	@printf "\n"

#════════════════════════════════════════════════════════════════════
# COMPLETE PIPELINES
#════════════════════════════════════════════════════════════════════

trocr-pipeline: install-trocr text-generate
	@$(call echo_header)
	@$(call echo_step,"Running full TROCR pipeline for: $(SCRIPTS)")
	@$(MAKE) ocr-process OCR_ENGINE=trocr SCRIPTS="$(SCRIPTS)"
	@$(MAKE) analyze OCR_ENGINE=trocr SCRIPTS="$(SCRIPTS)"
	@$(MAKE) validate OCR_ENGINE=trocr SCRIPTS="$(SCRIPTS)"
	@$(MAKE) report OCR_ENGINE=trocr SCRIPTS="$(SCRIPTS)"
	@$(call echo_footer)
	@$(MAKE) print-summary OCR_ENGINE=trocr SCRIPTS="$(SCRIPTS)"
print-summary:
	@$(call echo_header)
	@$(call echo_title,"$(shell echo $(OCR_ENGINE) | tr '[:lower:]' '[:upper:]') COMPLETE PIPELINE FINISHED")
	@$(call echo_footer)
	@printf "$(HIGHLIGHT)Pipeline stages completed:$(RESET)\n"
	@printf "  1. Text → Images: $(TEXT_IMAGES_DIR)\n"
	@printf "  2. $(OCR_ENGINE) Processing: $(OCR_OUTPUT_DIR)\n"
	@printf "  3. Spectral Analysis: $(ANALYSIS_OUTPUT_DIR)\n\n"
	@printf "$(HIGHLIGHT)Processed scripts:$(RESET)\n"
	@for script in $(SCRIPTS); do printf "  • $$script\n"; done
	@printf "\n$(HIGHLIGHT)Key outputs:$(RESET)\n"
	@printf "  Report:  $(ANALYSIS_OUTPUT_DIR)/report.md\n"
	@printf "  Figures: $(ANALYSIS_OUTPUT_DIR)/figures/\n"
	@printf "  Logs:    $(LOGS_DIR)/\n\n"

paddle-pipeline: install-paddle text-generate
	@$(call echo_header)
	@$(call echo_step,"Running full PADDLEOCR pipeline for: $(SCRIPTS)")
	@$(MAKE) ocr-process OCR_ENGINE=paddle SCRIPTS="$(SCRIPTS)"
	@$(MAKE) analyze OCR_ENGINE=paddle SCRIPTS="$(SCRIPTS)"
	@$(MAKE) validate OCR_ENGINE=paddle SCRIPTS="$(SCRIPTS)"
	@$(MAKE) report OCR_ENGINE=paddle SCRIPTS="$(SCRIPTS)"
	@$(call echo_footer)
	@$(MAKE) print-summary OCR_ENGINE=paddle SCRIPTS="$(SCRIPTS)"
	@$(call echo_header)
	@$(call echo_title,"✓ PADDLEOCR COMPLETE PIPELINE FINISHED")
	@$(call echo_footer)
	@printf "$(HIGHLIGHT)Pipeline stages completed:$(RESET)\n"
	@printf "  1. Text → Images: $(TEXT_IMAGES_DIR)\n"
	@printf "  2. $(OCR_ENGINE) Processing: $(OCR_OUTPUT_DIR)\n"
	@printf "  3. Spectral Analysis: $(ANALYSIS_OUTPUT_DIR)\n\n"
	@printf "$(HIGHLIGHT)Processed scripts:$(RESET)\n"
	@for script in $(SCRIPTS); do printf "  • $$script\n"; done
	@printf "\n$(HIGHLIGHT)Key outputs:$(RESET)\n"
	@printf "  Logs:    $(LOGS_DIR)/\n\n"

glm-pipeline: install-glm text-generate
	@$(call echo_header)
	@$(call echo_step,"Running full GLM-OCR pipeline for: $(SCRIPTS)")
	@$(MAKE) ocr-process OCR_ENGINE=glm SCRIPTS="$(SCRIPTS)"
	@$(MAKE) analyze OCR_ENGINE=glm SCRIPTS="$(SCRIPTS)"
	@$(MAKE) validate OCR_ENGINE=glm SCRIPTS="$(SCRIPTS)"
	@$(MAKE) report OCR_ENGINE=glm SCRIPTS="$(SCRIPTS)"
	@$(call echo_footer)
	@$(MAKE) print-summary OCR_ENGINE=glm SCRIPTS="$(SCRIPTS)"
	@$(call echo_header)
	@$(call echo_title,"✓ GLM-OCR COMPLETE PIPELINE FINISHED")
	@$(call echo_footer)
	@printf "$(HIGHLIGHT)Pipeline stages completed:$(RESET)\n"
	@printf "  1. Text → Images: $(TEXT_IMAGES_DIR)\n"
	@printf "  2. $(OCR_ENGINE) Processing: $(OCR_OUTPUT_DIR)\n"
	@printf "  3. Spectral Analysis: $(ANALYSIS_OUTPUT_DIR)\n\n"
	@printf "$(HIGHLIGHT)Processed scripts:$(RESET)\n"
	@for script in $(SCRIPTS); do printf "  • $$script\n"; done
	@printf "\n$(HIGHLIGHT)Key outputs:$(RESET)\n"
	@printf "  Report:  $(ANALYSIS_OUTPUT_DIR)/report.md\n"
	@printf "  Figures: $(ANALYSIS_OUTPUT_DIR)/figures/\n"
	@printf "  Logs:    $(LOGS_DIR)/\n\n"

surya-pipeline: install-surya text-generate
	@$(call echo_header)
	@$(call echo_step,"Running full SURYA pipeline for: $(SCRIPTS)")
	@$(MAKE) ocr-process OCR_ENGINE=surya SCRIPTS="$(SCRIPTS)"
	@$(MAKE) analyze OCR_ENGINE=surya SCRIPTS="$(SCRIPTS)"
	@$(MAKE) validate OCR_ENGINE=surya SCRIPTS="$(SCRIPTS)"
	@$(MAKE) report OCR_ENGINE=surya SCRIPTS="$(SCRIPTS)"
	@$(call echo_footer)
	@$(MAKE) print-summary OCR_ENGINE=surya SCRIPTS="$(SCRIPTS)"
	@$(call echo_header)
	@$(call echo_title,"✓ SURYA COMPLETE PIPELINE FINISHED")
	@$(call echo_footer)
	@printf "$(HIGHLIGHT)Pipeline stages completed:$(RESET)\n"
	@printf "  1. Text → Images: $(TEXT_IMAGES_DIR)\n"
	@printf "  2. $(OCR_ENGINE) Processing: $(OCR_OUTPUT_DIR)\n"
	@printf "  3. Spectral Analysis: $(ANALYSIS_OUTPUT_DIR)\n\n"
	@printf "$(HIGHLIGHT)Processed scripts:$(RESET)\n"
	@for script in $(SCRIPTS); do printf "  • $$script\n"; done
	@printf "\n$(HIGHLIGHT)Key outputs:$(RESET)\n"
	@printf "  Report:  $(ANALYSIS_OUTPUT_DIR)/report.md\n"
	@printf "  Figures: $(ANALYSIS_OUTPUT_DIR)/figures/\n"
	@printf "  Logs:    $(LOGS_DIR)/\n\n"

tesseract-pipeline: install-tesseract text-generate
	@$(call echo_header)
	@$(call echo_step,"Running full TESSERACT pipeline for: $(SCRIPTS)")
	@$(MAKE) ocr-process OCR_ENGINE=tesseract SCRIPTS="$(SCRIPTS)"
	@$(MAKE) analyze OCR_ENGINE=tesseract SCRIPTS="$(SCRIPTS)"
	@$(MAKE) validate OCR_ENGINE=tesseract SCRIPTS="$(SCRIPTS)"
	@$(MAKE) report OCR_ENGINE=tesseract SCRIPTS="$(SCRIPTS)"
	@$(call echo_footer)
	@$(MAKE) print-summary OCR_ENGINE=tesseract SCRIPTS="$(SCRIPTS)"
	@$(call echo_header)
	@$(call echo_title,"✓ TESSERACT COMPLETE PIPELINE FINISHED")
	@$(call echo_footer)
	@printf "$(HIGHLIGHT)Pipeline stages completed:$(RESET)\n"
	@printf "  1. Text → Images: $(TEXT_IMAGES_DIR)\n"
	@printf "  2. $(OCR_ENGINE) Processing: $(OCR_OUTPUT_DIR)\n"
	@printf "  3. Spectral Analysis: $(ANALYSIS_OUTPUT_DIR)\n\n"
	@printf "$(HIGHLIGHT)Processed scripts:$(RESET)\n"
	@for script in $(SCRIPTS); do printf "  • $$script\n"; done
	@printf "\n$(HIGHLIGHT)Key outputs:$(RESET)\n"
	@printf "  Report:  $(ANALYSIS_OUTPUT_DIR)/report.md\n"
	@printf "  Figures: $(ANALYSIS_OUTPUT_DIR)/figures/\n"
	@printf "  Logs:    $(LOGS_DIR)/\n\n"

easyocr-pipeline: install-easyocr text-generate
	@$(call echo_header)
	@$(call echo_step,"Running full EASYOCR pipeline for: $(SCRIPTS)")
	@$(call echo_warning,"Note: EasyOCR only supports Latin and Cyrillic")
	@$(MAKE) ocr-process OCR_ENGINE=easyocr SCRIPTS="$(SCRIPTS)"
	@$(MAKE) analyze OCR_ENGINE=easyocr SCRIPTS="$(SCRIPTS)"
	@$(MAKE) validate OCR_ENGINE=easyocr SCRIPTS="$(SCRIPTS)"
	@$(MAKE) report OCR_ENGINE=easyocr SCRIPTS="$(SCRIPTS)"
	@$(call echo_footer)
	@$(MAKE) print-summary OCR_ENGINE=easyocr SCRIPTS="$(SCRIPTS)"
	@$(call echo_header)
	@$(call echo_title,"✓ EASYOCR COMPLETE PIPELINE FINISHED")
	@$(call echo_footer)
	@printf "$(HIGHLIGHT)Pipeline stages completed:$(RESET)\n"
	@printf "  1. Text → Images: $(TEXT_IMAGES_DIR)\n"
	@printf "  2. $(OCR_ENGINE) Processing: $(OCR_OUTPUT_DIR)\n"
	@printf "  3. Spectral Analysis: $(ANALYSIS_OUTPUT_DIR)\n\n"
	@printf "$(HIGHLIGHT)Processed scripts:$(RESET)\n"
	@for script in $(SCRIPTS); do printf "  • $$script\n"; done
	@printf "\n$(HIGHLIGHT)Key outputs:$(RESET)\n"
	@printf "  Report:  $(ANALYSIS_OUTPUT_DIR)/report.md\n"
	@printf "  Figures: $(ANALYSIS_OUTPUT_DIR)/figures/\n"
	@printf "  Logs:    $(LOGS_DIR)/\n\n"

# Default "pipeline" alias: run TrOCR end-to-end
pipeline: trocr-pipeline
#════════════════════════════════════════════════════════════════════
# STATUS & DEBUGGING
#════════════════════════════════════════════════════════════════════

status:
	@$(call echo_header)
	@$(call echo_title,"PIPELINE STATUS")
	@$(call echo_footer)
	@printf "$(HIGHLIGHT)Stage 1: TEXT-TO-IMAGE$(RESET)\n"
	@if [ -d $(TEXT_IMAGES_DIR) ] && [ -n "$$(find $(TEXT_IMAGES_DIR) -name '*.png' 2>/dev/null)" ]; then \
		printf "  $(SUCCESS)✓ Images generated$(RESET)\n"; \
		for script in $(SCRIPTS); do \
			count=$$(ls -1 $(TEXT_IMAGES_DIR)/$$script/*.png 2>/dev/null | wc -l); \
			printf "    $$script: $$count images\n"; \
		done; \
	else \
		printf "  $(ERROR)✗ No images generated$(RESET)\n"; \
	fi
	@printf "\n$(HIGHLIGHT)Stage 2: OCR$(RESET)\n"
	@if [ -d $(OCR_OUTPUT_DIR) ] && [ -n "$$(ls -A $(OCR_OUTPUT_DIR)/*.npz 2>/dev/null)" ]; then \
		printf "  $(SUCCESS)✓ Confusion matrices generated$(RESET)\n"; \
		ls -1 $(OCR_OUTPUT_DIR)/*.npz 2>/dev/null | xargs -I {} basename {} .npz | sed 's/^/    /'; \
	else \
		printf "  $(ERROR)✗ No confusion matrices$(RESET)\n"; \
	fi
	@printf "\n$(HIGHLIGHT)Stage 3: ANALYSIS$(RESET)\n"
	@if [ -f $(ANALYSIS_OUTPUT_DIR)/results.json ]; then \
		printf "  $(SUCCESS)✓ Analysis complete$(RESET)\n"; \
		printf "    Results: $(ANALYSIS_OUTPUT_DIR)/results.json\n"; \
		printf "    Figures: $$(ls -1 $(ANALYSIS_OUTPUT_DIR)/figures/*.png 2>/dev/null | wc -l) files\n"; \
	else \
		printf "  $(ERROR)✗ Analysis not run$(RESET)\n"; \
	fi
	@printf "\n$(HIGHLIGHT)Logs:$(RESET)\n"
	@if [ -d $(LOGS_DIR) ] && [ -n "$$(ls -A $(LOGS_DIR) 2>/dev/null)" ]; then \
		ls -1 $(LOGS_DIR) | sed 's/^/  /'; \
	else \
		printf "  (no logs yet)\n"; \
	fi
	@printf "\n"

summary:
	@$(call echo_header)
	@$(call echo_title,"RESULTS SUMMARY")
	@$(call echo_footer)
	@if [ -f $(ANALYSIS_OUTPUT_DIR)/results.json ]; then \
		$(PYTHON) -m json.tool $(ANALYSIS_OUTPUT_DIR)/results.json 2>/dev/null | head -40; \
		printf "  ...\n"; \
	else \
		$(call echo_warning,"No results found - run 'make analyze' first"); \
	fi
	@printf "\n"

logs:
	@$(call echo_step,"Recent logs")
	@if [ -d $(LOGS_DIR) ] && [ -n "$$(ls -A $(LOGS_DIR) 2>/dev/null)" ]; then \
		ls -1 $(LOGS_DIR) | sed 's/^/  /'; \
	else \
		printf "  (no logs yet)\n"; \
	fi
	@printf "\n"

debug: check-env
	@$(call echo_header)
	@$(call echo_title,"DEBUG INFORMATION")
	@$(call echo_footer)
	@printf "$(BOLD)Directories:$(RESET)\n"
	@printf "  Text input:       $(TEXT_INPUT_DIR) - $$([ -d $(TEXT_INPUT_DIR) ] && echo EXISTS || echo MISSING)\n"
	@printf "  Images:           $(TEXT_IMAGES_DIR) - $$([ -d $(TEXT_IMAGES_DIR) ] && echo EXISTS || echo MISSING)\n"
	@printf "  Ground truth:     $(TEXT_GT_DIR) - $$([ -d $(TEXT_GT_DIR) ] && echo EXISTS || echo MISSING)\n"
	@printf "  OCR output:       $(OCR_OUTPUT_DIR) - $$([ -d $(OCR_OUTPUT_DIR) ] && echo EXISTS || echo MISSING)\n"
	@printf "  Analysis output:  $(ANALYSIS_OUTPUT_DIR) - $$([ -d $(ANALYSIS_OUTPUT_DIR) ] && echo EXISTS || echo MISSING)\n\n"
	@printf "$(BOLD)Pipeline configuration:$(RESET)\n"
	@printf "  Scripts:          $(SCRIPTS)\n"
	@printf "  OCR Engine:       $(OCR_ENGINE)\n"
	@printf "  Device:           $(OCR_DEVICE)\n\n"
	@printf "$(BOLD)Python environment:$(RESET)\n"
	@printf "  Python: $$($(PYTHON) --version 2>&1)\n"
	@printf "  Pip:    $$($(PIP) --version 2>&1 | head -1)\n\n"
	@printf "$(BOLD)Key packages:$(RESET)\n"
	@$(PYTHON) -c "import numpy, pandas, scipy, matplotlib, seaborn; print(f'  NumPy: {numpy.__version__}'); print(f'  Pandas: {pandas.__version__}'); print(f'  SciPy: {scipy.__version__}'); print(f'  Matplotlib: {matplotlib.__version__}'); print(f'  Seaborn: {seaborn.__version__}')" 2>/dev/null || $(call echo_warning,"Could not verify package versions")
	@printf "\n"

debug-trocr: preflight
	@$(call echo_header)
	@$(call echo_title,"DEBUGGING TROCR")
	@$(call echo_footer)
	@$(call echo_step,"Running TrOCR debug test")
	@$(PYTHON) src/spectral_scripts/ocr_pipeline/debug_trocr.py

debug-ocr: preflight
	@$(call echo_header)
	@$(call echo_title,"DEBUGGING OCR OUTPUT")
	@$(call echo_footer)
	@$(call echo_step,"Analyzing OCR pipeline output")
	@$(PYTHON) src/spectral_scripts/ocr_pipeline/debug_ocr_output.py

debug-all: debug debug-trocr debug-ocr
	@$(call echo_header)
	@$(call echo_title,"✓ FULL DEBUG COMPLETE")
	@$(call echo_footer)

#════════════════════════════════════════════════════════════════════
# TESTING & QUALITY
#════════════════════════════════════════════════════════════════════

test: preflight | $(LOGS_DIR)
	@$(call echo_header)
	@$(call echo_title,"RUNNING UNIT TESTS")
	@$(call echo_footer)
	@$(call echo_step,"Executing pytest")
	@$(PYTEST) -s 2>&1 | tee $(LOGS_DIR)/tests.log
	@$(call echo_success,"Tests completed\n")

lint:
	@$(call echo_header)
	@$(call echo_title,"RUNNING LINT CHECKS")
	@$(call echo_footer)
	@$(call echo_step,"Executing ruff")
	@ruff check .
	@$(call echo_success,"Lint checks passed\n")

#════════════════════════════════════════════════════════════════════
# CLEANUP
#════════════════════════════════════════════════════════════════════

clean:
	@$(call echo_header)
	@$(call echo_title,"CLEANING OUTPUTS")
	@$(call echo_footer)
	@$(call echo_step,"Removing analysis outputs only")
	@rm -f $(ANALYSIS_OUTPUT_DIR)/*.json
	@rm -f $(ANALYSIS_OUTPUT_DIR)/report.md
	@rm -rf $(ANALYSIS_OUTPUT_DIR)/figures
	@rm -rf $(ANALYSIS_OUTPUT_DIR)/validation
	@$(call echo_success,"Analysis outputs cleaned")
	@printf "  $(HIGHLIGHT)Kept:$(RESET) Text images, OCR matrices, logs\n\n"

distclean: clean
	@$(call echo_header)
	@$(call echo_title,"FULL CLEANUP")
	@$(call echo_footer)
	@$(call echo_step,"Removing ALL generated files")
	@rm -rf $(TEXT_IMAGES_DIR)/*/*.png
	@rm -rf $(TEXT_GT_DIR)/*.txt
	@rm -f $(OCR_OUTPUT_DIR)/*.npz
	@rm -f $(OCR_OUTPUT_DIR)/*.json
	@rm -f $(OCR_OUTPUT_DIR)/*.md
	@rm -rf $(LOGS_DIR)/*
	@$(call echo_success,"Full cleanup complete")
	@printf "  $(HIGHLIGHT)Kept:$(RESET) Text input files, project structure\n\n"

#════════════════════════════════════════════════════════════════════
# DEFAULT TARGET
#════════════════════════════════════════════════════════════════════

.DEFAULT_GOAL := help

#════════════════════════════════════════════════════════════════════
# END OF MAKEFILE
#════════════════════════════════════════════════════════════════════