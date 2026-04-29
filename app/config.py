# config.py

# Maximum depth for the hierarchical chapter/topic picker.
#   1 = H1 only
#   2 = H1 + H2  (chapters + sections)          <-- single source default
#   3 = source + chapters + sections             <-- recommended with multiple books
#
# When markdown files include a `source:` YAML frontmatter field (written by
# scrape_to_md.py), the breadcrumb gains an extra top-level segment, e.g.:
#   "Deep Learning with Python > Chapter 1: ... > Artificial intelligence"
# Set this to 3 when mixing multiple books or other source types.
TOPIC_TREE_MAX_DEPTH = 3

# Score threshold (out of 5) for counting a practice answer as correct.
FEEDBACK_PASS_THRESHOLD = 3