#!/bin/bash
# demo_m1.sh - Proves M1 works end-to-end

set -e  # Exit on error

echo "=== Vestig M1 Demo ==="
echo ""

echo "Step 1: Adding memories..."
echo ""

# Capture first ID for later inspection
FIRST_ID=$(vestig memory add "Solved authentication bug by checking JWT token expiry in middleware" | grep -oE 'mem_[a-f0-9-]+' | head -1)

vestig memory add "User prefers dark mode and minimal UI"
vestig memory add "Fixed database migration error by running migrations in correct order"
vestig memory add "Learned that Redis cache invalidation needs explicit TTL settings"
vestig memory add "User gave positive feedback on fast API response times"
vestig memory add "Debugging tip: always check logs before assuming code issue"
vestig memory add "React useState hook caused infinite loop when missing dependency array"
vestig memory add "User likes to work in short focused sprints, not long sessions"
vestig memory add "PostgreSQL index on user_id column improved query speed 10x"
vestig memory add "Error handling: always return user-friendly messages, not stack traces"

echo ""
echo "Step 2: Testing retrieval..."
echo ""

echo "Query 1: 'authentication problems'"
vestig memory search "authentication problems" --limit 3
echo ""

echo "Query 2: 'database performance'"
vestig memory search "database performance" --limit 3
echo ""

echo "Query 3: 'user preferences'"
vestig memory search "user preferences" --limit 3
echo ""

echo "Query 4: 'React debugging' (formatted for agent)"
vestig memory recall "React debugging"
echo ""

echo "Step 3: Inspecting a memory..."
echo ""
echo "Showing first memory added (ID: $FIRST_ID):"
vestig memory show "$FIRST_ID"
echo ""

echo "=== Demo Complete ==="
echo "Run this script again to verify persistence across sessions."
