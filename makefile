# Define source directory
SRC_DIR := posts

# Find all Markdown files in the source directory
MARKDOWN_FILES := $(wildcard $(SRC_DIR)/*.md)

# Define the corresponding HTML files in the same directory
HTML_FILES := $(MARKDOWN_FILES:.md=.html)

# Add index.html to the list of files to process
HTML_FILES += index.html

# Default target
all: $(HTML_FILES)

# Rule for converting Markdown files in the posts directory to HTML
$(SRC_DIR)/%.html: $(SRC_DIR)/%.md template.html
	pandoc -s --template=template.html -c ../style.css $< -o $@

# Rule for converting index.md in the root directory
index.html: index.md template.html
	pandoc -s --template=template.html -c style.css $< -o $@

# Clean target
clean:
	rm -f $(HTML_FILES)

# Serve target to run Python HTTP server
serve: all
	@echo "Starting Python HTTP server on http://localhost:8000"
	@python3 -m http.server 8000 || python -m SimpleHTTPServer 8000

.PHONY: all clean serve
