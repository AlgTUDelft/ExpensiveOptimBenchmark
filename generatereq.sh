# Create a frozen list of requirements. Also dependent on extras potentially.
poetry export -f requirements.txt > requirements.txt
poetry export -E smac -f requirements.txt > requirements_smac.txt