# Milestone 1: Fork Setup & Dependencies

## Objective
Fork wrale/mcp-server-tree-sitter, establish rickbarraza/mcp-server-project-memory, install all dependencies needed for entire project.

## Actions Required

### Repository Setup
1. Fork `wrale/mcp-server-tree-sitter` to `rickbarraza/mcp-server-project-memory`
2. Clone locally: `git clone https://github.com/rickbarraza/mcp-server-project-memory.git`
3. Navigate to directory: `cd mcp-server-project-memory`

### Baseline Verification
```bash
npm install
npm run build
npm test
```

### Install All Project Dependencies
```bash
# Memory and embedding dependencies
npm install --save chromadb sentence-transformers-js hash-wasm chokidar

# Development dependencies  
npm install --save-dev @types/node @types/chokidar

# Verify installation
npm list --depth=0
```

### Identity Updates
1. Update `package.json`:
   - Change name to "mcp-server-project-memory"
   - Update description to include "semantic memory and RAG capabilities"
   - Add Rick Barraza as contributor

2. Create `ENHANCEMENT_PLAN.md` with:
   ```markdown
   # Project Memory Enhancement Plan
   
   ## Base Repository
   Forked from: wrale/mcp-server-tree-sitter
   
   ## Enhancements Added
   - [ ] Milestone 1: Fork setup & dependencies ✓
   - [ ] Milestone 2: Memory infrastructure 
   - [ ] Milestone 3: MCP tool extensions
   - [ ] Milestone 4: Integration & testing
   
   ## Status
   Current milestone: 1 (COMPLETE)
   ```

3. Update README.md header:
   ```markdown
   # MCP Server: Project Memory
   
   A Model Context Protocol server providing code analysis with semantic memory capabilities.
   Based on wrale/mcp-server-tree-sitter with RAG enhancements.
   ```

## Test Conditions - Milestone 1 COMPLETE When:
- [ ] Repository successfully forked to rickbarraza/mcp-server-project-memory
- [ ] All dependencies installed without errors: `npm list chromadb sentence-transformers-js hash-wasm chokidar`
- [ ] Baseline build passes: `npm run build` exits with code 0
- [ ] Baseline tests pass: `npm test` exits with code 0  
- [ ] package.json name field shows "mcp-server-project-memory"
- [ ] ENHANCEMENT_PLAN.md exists with Milestone 1 checked as complete
- [ ] README.md updated with new identity

## On Completion
Update ENHANCEMENT_PLAN.md to mark Milestone 1 as ✓ and ready for Milestone 2.