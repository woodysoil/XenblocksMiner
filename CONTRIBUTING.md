# Contributing to XenBlocks Mining Platform

## Development Setup

### Prerequisites

- **Node.js** >= 18 and **npm** >= 9
- **Python** >= 3.10
- Git

### Backend

```bash
cd server
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m server.server          # MQTT :1883 + REST API :8080
```

### Frontend

```bash
cd web
npm install
npm run dev                      # Vite dev server :5173
```

### Full Stack

Start the backend first, then the frontend. The dashboard is also served at `http://localhost:8080/dashboard`.

---

## Code Style

### TypeScript (Frontend)

- **Strict mode** enabled (`tsconfig.json` — `"strict": true`).
- Target **ES2020**, JSX via `react-jsx`.
- Use **named exports** for components; use `export default` only for page-level components (lazy-loaded).
- Prefer `const` over `let`. Never use `var`.
- Destructure props in function signatures.

### Python (Backend)

- Type hints on all public function signatures.
- Use `async def` for any I/O-bound endpoint or service method.
- Pydantic v2 models for request/response schemas.
- Imports ordered: stdlib, third-party, project (`server.*`).

### Tailwind CSS

- **Tailwind v4** via `@tailwindcss/vite` plugin — no `tailwind.config.js` needed.
- Use arbitrary values with bracket notation (`bg-[#141820]`) for design-token colors.
- Always use token-defined colors from `src/design/tokens.ts` — never invent new hex values.
- Keep class strings on a single line when under ~120 chars; break onto multiple lines for long lists.

---

## Design System Rules

All visual constants live in `web/src/design/tokens.ts`. This file is the single source of truth.

### When to Use What

| Need | Use | Do NOT |
|---|---|---|
| Color, spacing, radius, font | Import from `tokens.ts` (`colors`, `space`, `radius`, `font`) | Hardcode hex values in components |
| Common UI patterns (card, badge, button, input) | `tw.*` presets from `tokens.ts` | Duplicate long class strings |
| Chart styling | `chartTheme` from `tokens.ts` | Inline chart style objects |

### Component Creation Guidelines

1. Create the file in `web/src/design/` (e.g., `MyComponent.tsx`).
2. Use a default export: `export default function MyComponent(props: MyComponentProps) { ... }`.
3. Define a `Props` interface and export it if consumers need it.
4. Import colors/spacing only from `tokens.ts` — reference `tw.*` presets or `colors.*` values.
5. Re-export from `web/src/design/index.ts`:
   ```ts
   export { default as MyComponent } from './MyComponent';
   ```
6. Consumers import from `@/design` (or `../design`), never from the file directly.

### Tailwind Purge Safety

Because Tailwind v4 scans source files for class usage, all token-based classes use bracket notation (`bg-[#141820]`). This is safe — bracket values are always included. Do **not** dynamically construct class names with template literals across token boundaries (e.g., `` `bg-[${color}]` ``). Instead, map to complete class strings.

---

## Git Workflow

### Branch Naming

```
feat/short-description
fix/short-description
refactor/short-description
docs/short-description
test/short-description
perf/short-description
```

### Commit Messages

Follow the format used in this repository:

```
<type>: <concise description>
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `perf`, `polish`.

Examples from this repo:
- `feat: hashpower marketplace — monitoring dashboard, modular backend, design system`
- `fix: Provider page data bug and add pagination test script`
- `perf: optimize dashboard backend — COUNT queries, N+1 fix, pagination`
- `refactor: architect audit fixes — code split, a11y, skeletons, shared utils`

### PR Process

1. Create a branch from `main`.
2. Make focused, atomic commits.
3. Ensure `npm run build` passes (frontend) and `pytest tests/` passes (backend).
4. Open a PR against `main`. Describe what changed and why.
5. Address review feedback, then squash-merge.

---

## Testing

### Backend

```bash
# All tests
pytest tests/

# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Single test file
pytest tests/unit/test_monitoring.py -v
```

Tests live in `tests/unit/` and `tests/integration/`. Integration tests use `conftest.py` fixtures that stand up the server components in-process.

### Frontend

No test runner is configured. Build verification serves as the gate:

```bash
cd web
npm run build    # runs tsc -b && vite build
```

A successful build confirms type-safety and bundle correctness.

---

## Adding New Pages

Checklist:

- [ ] Create `web/src/pages/MyPage.tsx` with a default export.
- [ ] Add a lazy import in `web/src/App.tsx`:
  ```tsx
  const MyPage = lazy(() => import("./pages/MyPage"));
  ```
- [ ] Add a `<Route>` inside the `<Route element={<Layout />}>` block:
  ```tsx
  <Route path="my-page" element={<MyPage />} />
  ```
- [ ] Add a nav entry in `web/src/components/Layout.tsx` — append to `publicNav` or `walletNav`:
  ```ts
  { to: "/my-page", label: "My Page", icon: (<svg>...</svg>) },
  ```
- [ ] Add the page title in the `pageTitles` map in `Layout.tsx`:
  ```ts
  "/my-page": "My Page",
  ```
- [ ] The `*` catch-all route already renders `<NotFound />` — no action needed for 404.

---

## Adding New API Endpoints

Checklist:

- [ ] Create `server/routers/my_feature.py` with a `router = APIRouter(prefix="/api/my-feature", tags=["my-feature"])`.
- [ ] Access shared services via `request.app.state.server` (the `PlatformServer` instance):
  ```python
  from fastapi import APIRouter, Request

  router = APIRouter(prefix="/api/my-feature", tags=["my-feature"])

  @router.get("/")
  async def list_items(request: Request):
      server = request.app.state.server
      # use server.storage, server.accounts, etc.
  ```
- [ ] Register the router in `server/routers/__init__.py`:
  ```python
  from server.routers import my_feature
  # inside register_all_routers():
  app.include_router(my_feature.router)
  ```
- [ ] Add request/response Pydantic models in `server/models.py` or inline in the router file.
- [ ] Write tests in `tests/integration/` or `tests/unit/`.

---

## Adding New Components

Checklist:

- [ ] Create `web/src/design/MyComponent.tsx`.
- [ ] Define and export a `Props` interface.
- [ ] Use tokens (`colors`, `tw`, `space`, `radius`) — no hardcoded values.
- [ ] Use a default export.
- [ ] Re-export from `web/src/design/index.ts`:
  ```ts
  export { default as MyComponent } from './MyComponent';
  ```
- [ ] If the component is page-specific and not reusable, place it in the page file or `web/src/components/` instead — the `design/` directory is reserved for the shared design system.

---

## Code Review Checklist

Reviewers verify the following before approving:

- [ ] **Build passes** — `npm run build` (frontend), `pytest tests/` (backend).
- [ ] **No hardcoded colors/spacing** — all visual values come from `tokens.ts`.
- [ ] **Design system components re-exported** — new `design/` components appear in `index.ts`.
- [ ] **Type safety** — no `any` casts without justification; Pydantic models for API schemas.
- [ ] **Lazy loading** — new pages use `lazy(() => import(...))` in `App.tsx`.
- [ ] **Route + nav consistency** — new pages have matching route, nav entry, and page title.
- [ ] **Router registered** — new API routers are included in `routers/__init__.py`.
- [ ] **No regressions** — existing tests still pass; no unrelated files modified.
- [ ] **Commit hygiene** — messages follow `type: description` format; changes are atomic.
