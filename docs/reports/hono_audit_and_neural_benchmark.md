# Comprehensive Code Oracle Audit & Neural Benchmark Report: Hono

- **Target Repository:** `honojs/hono` (/home/wxsys/code-oracle/benchmarks_repos/hono)
- **Audit Date:** 2026-09-25 08:10:09 UTC
- **Evaluator Engine:** Code Oracle v0.1.0
- **Language:** TypeScript & JavaScript (100% Tree-sitter & Inverted Index Map)
- **Neural Model:** Laya ModernBERT-base 164M (BF16, 312 MB, `/home/wxsys/code-oracle/weights_base`)

---

## Executive Summary

Code Oracle was deployed against the complete TypeScript/JavaScript codebase of **Hono** (`/home/wxsys/code-oracle/benchmarks_repos/hono`).
Across **360 files**, the neuro-symbolic engine indexed **1715 symbols**, inspected **62230 call sites**, and resolved **3886 import edges**.

| Audit Domain | Findings Count | Execution Latency | Verdict Summary |
| :--- | :---: | :---: | :--- |
| **1. Workspace Topology Indexing** | 1715 symbols | 446.18 ms | Complete TS/JS AST map resolved |
| **2. Architectural Import Cycles** | 5 cycles | 12.30 ms | Core compose/context cycle + JSX cycle detected |
| **3. Performance & Blocking Diagnostics** | 62 diagnostics | 2048.80 ms | 9 sync I/O in async, 46 nested loops, 7 loop queries |
| **4. Dead Code Reachability** | 163 symbols (1169 lines) | 1212.33 ms | 163 orphan adapters, interfaces, and handlers |
| **5. Neuro-Symbolic Verification** | 4 Patch Scenarios | 0.9 - 890 ms (CPU) | 100% precision on clean vs syntax vs arity vs cycle |

---

## 1. Architectural Issues: Circular Dependency Cycles

Tarjan Strongly Connected Components (SCC) detected **5 distinct import cycles** in Hono's codebase:

### Cycle 1: 5 Modules
```
  src/compose.ts ->
  src/hono-base.ts ->
  src/types.ts ->
  src/request.ts ->
  src/context.ts ->
  src/compose.ts [CYCLE]
```

### Cycle 2: 2 Modules
```
  src/helper/ssg/plugins.ts ->
  src/helper/ssg/ssg.ts ->
  src/helper/ssg/plugins.ts [CYCLE]
```

### Cycle 3: 2 Modules
```
  src/helper/streaming/index.ts ->
  src/helper/streaming/text.ts ->
  src/helper/streaming/index.ts [CYCLE]
```

### Cycle 4: 7 Modules
```
  src/jsx/base.ts ->
  src/jsx/types.ts ->
  src/jsx/streaming.ts ->
  src/jsx/components.ts ->
  src/jsx/children.ts ->
  src/jsx/index.ts ->
  src/jsx/context.ts ->
  src/jsx/base.ts [CYCLE]
```

### Cycle 5: 2 Modules
```
  src/utils/jwt/jws.ts ->
  src/utils/jwt/types.ts ->
  src/utils/jwt/jws.ts [CYCLE]
```

## 2. Performance Anti-Patterns & Blocking Calls (`perf-lint`)

Total Diagnostics: **62** (Errors: `17`, Warnings: `45`)

- `PERF004` (Blocking Synchronous Call in Async Function): **9 occurrences**
- `PERF001` (Nested Loop Complexity $O(N^2)/O(N^3)/O(N^4)$): **46 occurrences**
- `PERF002` (I/O or Query Call Inside Loop Body): **7 occurrences**

| Rule | Severity | Location | Line | Details |
| :--- | :---: | :--- | :---: | :--- |
| `PERF004` | **error** | `benchmarks/http-server/benchmark.ts` | 121 | Blocking synchronous call 'mkdirSync' inside async function 'buildVersion' |
| `PERF004` | **error** | `benchmarks/http-server/benchmark.ts` | 125 | Blocking synchronous call 'writeFileSync' inside async function 'buildVersion' |
| `PERF001` | **warn** | `benchmarks/http-server/benchmark.ts` | 204 | Nested loop complexity O(N^2) detected at depth 2 |
| `PERF004` | **error** | `benchmarks/http-server/benchmark.ts` | 315 | Blocking synchronous call 'writeFileSync' inside async function 'main' |
| `PERF004` | **error** | `benchmarks/http-server/benchmark.ts` | 320 | Blocking synchronous call 'existsSync' inside async function 'main' |
| `PERF004` | **error** | `benchmarks/http-server/benchmark.ts` | 321 | Blocking synchronous call 'rmSync' inside async function 'main' |
| `PERF001` | **warn** | `benchmarks/jsx-dom/benchmark.ts` | 8 | Nested loop complexity O(N^2) detected at depth 2 |
| `PERF001` | **error** | `benchmarks/jsx-dom/benchmark.ts` | 34 | Nested loop complexity O(N^3) detected at depth 3 |
| `PERF001` | **error** | `benchmarks/jsx-dom/benchmark.ts` | 43 | Nested loop complexity O(N^4) detected at depth 4 |
| `PERF001` | **error** | `benchmarks/jsx-dom/benchmark.ts` | 48 | Nested loop complexity O(N^4) detected at depth 4 |
| `PERF004` | **error** | `perf-measures/bundle-check/scripts/check-bundle-size.ts` | 20 | Blocking synchronous call 'fs.statSync' inside async function 'main' |
| `PERF004` | **error** | `perf-measures/bundle-check/scripts/check-bundle-size.ts` | 46 | Blocking synchronous call 'fs.existsSync' inside async function 'main' |
| `PERF004` | **error** | `perf-measures/bundle-check/scripts/check-bundle-size.ts` | 47 | Blocking synchronous call 'fs.unlinkSync' inside async function 'main' |
| `PERF001` | **warn** | `src/client/client.ts` | 76 | Nested loop complexity O(N^2) detected at depth 2 |
| `PERF001` | **warn** | `src/client/utils.ts` | 35 | Nested loop complexity O(N^2) detected at depth 2 |
| `PERF001` | **warn** | `src/context.ts` | 424 | Nested loop complexity O(N^2) detected at depth 2 |
| `PERF001` | **warn** | `src/context.ts` | 643 | Nested loop complexity O(N^2) detected at depth 2 |
| `PERF002` | **warn** | `src/helper/accepts/accepts.ts` | 66 | Possible N+1 query: I/O or database call 'supports.find' detected inside loop |
| `PERF001` | **warn** | `src/helper/css/common.ts` | 158 | Nested loop complexity O(N^2) detected at depth 2 |
| `PERF001` | **warn** | `src/helper/html/index.ts` | 23 | Nested loop complexity O(N^2) detected at depth 2 |
| `PERF001` | **warn** | `src/helper/ssg/ssg.test.tsx` | 30 | Nested loop complexity O(N^2) detected at depth 2 |
| `PERF001` | **warn** | `src/hono-base.ts` | 149 | Nested loop complexity O(N^2) detected at depth 2 |
| `PERF001` | **error** | `src/hono-base.ts` | 151 | Nested loop complexity O(N^3) detected at depth 3 |
| `PERF001` | **warn** | `src/jsx/dom/css.ts` | 39 | Nested loop complexity O(N^2) detected at depth 2 |
| `PERF004` | **error** | `src/jsx/dom/index.test.tsx` | 3319 | Blocking synchronous call 'flushSync' inside async function '<anonymous>' |
| `PERF001` | **warn** | `src/jsx/dom/intrinsic-element/components.ts` | 91 | Nested loop complexity O(N^2) detected at depth 2 |
| `PERF001` | **warn** | `src/jsx/dom/render.ts` | 651 | Nested loop complexity O(N^2) detected at depth 2 |
| `PERF001` | **warn** | `src/jsx/intrinsic-element/components.ts` | 47 | Nested loop complexity O(N^2) detected at depth 2 |
| `PERF002` | **warn** | `src/middleware/compress/index.ts` | 35 | Possible N+1 query: I/O or database call 'accepts.find' detected inside loop |
| `PERF001` | **warn** | `src/middleware/etag/digest.ts` | 42 | Nested loop complexity O(N^2) detected at depth 2 |
| `PERF001` | **warn** | `src/middleware/method-not-allowed/index.ts` | 122 | Nested loop complexity O(N^2) detected at depth 2 |
| `PERF001` | **warn** | `src/router/linear-router/router.ts` | 49 | Nested loop complexity O(N^2) detected at depth 2 |
| `PERF001` | **warn** | `src/router/linear-router/router.ts` | 82 | Nested loop complexity O(N^2) detected at depth 2 |
| `PERF002` | **warn** | `src/router/linear-router/router.ts` | 104 | Possible N+1 query: I/O or database call 'new RegExp(pattern, 'd').exec' detected inside loop |
| `PERF002` | **warn** | `src/router/pattern-router/router.ts` | 45 | Possible N+1 query: I/O or database call 'pattern.exec' detected inside loop |
| `PERF001` | **warn** | `src/router/reg-exp-router/node.ts` | 97 | Nested loop complexity O(N^2) detected at depth 2 |
| `PERF001` | **warn** | `src/router/reg-exp-router/node.ts` | 117 | Nested loop complexity O(N^2) detected at depth 2 |
| `PERF001` | **warn** | `src/router/reg-exp-router/prepared-router.ts` | 79 | Nested loop complexity O(N^2) detected at depth 2 |
| `PERF001` | **warn** | `src/router/reg-exp-router/prepared-router.ts` | 133 | Nested loop complexity O(N^2) detected at depth 2 |
| `PERF001` | **warn** | `src/router/reg-exp-router/router.test.ts` | 167 | Nested loop complexity O(N^2) detected at depth 2 |
| `PERF001` | **warn** | `src/router/reg-exp-router/router.test.ts` | 207 | Nested loop complexity O(N^2) detected at depth 2 |
| `PERF001` | **warn** | `src/router/reg-exp-router/router.test.ts` | 221 | Nested loop complexity O(N^2) detected at depth 2 |
| `PERF001` | **warn** | `src/router/reg-exp-router/router.test.ts` | 257 | Nested loop complexity O(N^2) detected at depth 2 |
| `PERF001` | **warn** | `src/router/reg-exp-router/router.ts` | 80 | Nested loop complexity O(N^2) detected at depth 2 |
| `PERF001` | **warn** | `src/router/reg-exp-router/router.ts` | 104 | Nested loop complexity O(N^2) detected at depth 2 |
| `PERF001` | **error** | `src/router/reg-exp-router/router.ts` | 105 | Nested loop complexity O(N^3) detected at depth 3 |
| `PERF001` | **warn** | `src/router/reg-exp-router/router.ts` | 116 | Nested loop complexity O(N^2) detected at depth 2 |
| `PERF001` | **warn** | `src/router/reg-exp-router/router.ts` | 155 | Nested loop complexity O(N^2) detected at depth 2 |
| `PERF001` | **warn** | `src/router/reg-exp-router/trie.ts` | 47 | Nested loop complexity O(N^2) detected at depth 2 |
| `PERF001` | **warn** | `src/router/smart-router/router.ts` | 35 | Nested loop complexity O(N^2) detected at depth 2 |
| `PERF001` | **warn** | `src/router/trie-router/node.ts` | 77 | Nested loop complexity O(N^2) detected at depth 2 |
| `PERF001` | **warn** | `src/router/trie-router/node.ts` | 104 | Nested loop complexity O(N^2) detected at depth 2 |
| `PERF001` | **error** | `src/router/trie-router/node.ts` | 121 | Nested loop complexity O(N^3) detected at depth 3 |
| `PERF001` | **error** | `src/router/trie-router/node.ts` | 149 | Nested loop complexity O(N^4) detected at depth 4 |
| `PERF002` | **warn** | `src/router/trie-router/node.ts` | 156 | Possible N+1 query: I/O or database call 'matcher.exec' detected inside loop |
| `PERF001` | **error** | `src/router/trie-router/node.ts` | 172 | Nested loop complexity O(N^4) detected at depth 4 |
| `PERF001` | **warn** | `src/utils/ipaddr.ts` | 31 | Nested loop complexity O(N^2) detected at depth 2 |
| `PERF001` | **warn** | `src/utils/ipaddr.ts` | 200 | Nested loop complexity O(N^2) detected at depth 2 |
| `PERF001` | **warn** | `src/utils/ipaddr.ts` | 216 | Nested loop complexity O(N^2) detected at depth 2 |
| `PERF001` | **warn** | `src/utils/url.ts` | 39 | Nested loop complexity O(N^2) detected at depth 2 |
| `PERF002` | **warn** | `src/validator/validator.test.ts` | 449 | Possible N+1 query: I/O or database call 'app.post' detected inside loop |
| `PERF002` | **warn** | `src/validator/validator.test.ts` | 490 | Possible N+1 query: I/O or database call 'app.post' detected inside loop |

## 3. Dead Code & Orphan Symbols (`dead-code`)

Total Dead Symbols: **163** (1169 total lines of code)

| Symbol | Kind | Location | Lines | Classification | Confidence |
| :--- | :---: | :--- | :---: | :--- | :---: |
| `LatticeProxyEventV2` | `interface` | `src/adapter/aws-lambda/handler.ts:30` | 10 | **Direct Orphan** | 1.00 |
| `APIGatewayProxyEventV2` | `interface` | `src/adapter/aws-lambda/handler.ts:42` | 21 | **Direct Orphan** | 1.00 |
| `APIGatewayProxyEvent` | `interface` | `src/adapter/aws-lambda/handler.ts:65` | 19 | **Direct Orphan** | 1.00 |
| `ALBProxyEvent` | `interface` | `src/adapter/aws-lambda/handler.ts:86` | 13 | **Direct Orphan** | 1.00 |
| `EventV2Processor` | `class` | `src/adapter/aws-lambda/handler.ts:408` | 36 | **Direct Orphan** | 1.00 |
| `EventV1Processor` | `class` | `src/adapter/aws-lambda/handler.ts:447` | 57 | **Direct Orphan** | 1.00 |
| `ALBProcessor` | `class` | `src/adapter/aws-lambda/handler.ts:507` | 78 | **Direct Orphan** | 1.00 |
| `LatticeV2Processor` | `class` | `src/adapter/aws-lambda/handler.ts:588` | 38 | **Direct Orphan** | 1.00 |
| `CognitoIdentity` | `interface` | `src/adapter/aws-lambda/types.ts:3` | 4 | **Direct Orphan** | 1.00 |
| `ClientContext` | `interface` | `src/adapter/aws-lambda/types.ts:8` | 6 | **Direct Orphan** | 1.00 |
| `ClientContextClient` | `interface` | `src/adapter/aws-lambda/types.ts:15` | 7 | **Direct Orphan** | 1.00 |
| `ClientContextEnv` | `interface` | `src/adapter/aws-lambda/types.ts:23` | 7 | **Direct Orphan** | 1.00 |
| `LatticeRequestContextV2` | `interface` | `src/adapter/aws-lambda/types.ts:155` | 19 | **Direct Orphan** | 1.00 |
| `BunServerWebSocket` | `interface` | `src/adapter/bun/websocket.ts:8` | 6 | **Direct Orphan** | 1.00 |
| `ServeStaticOptions` | `type_alias` | `src/adapter/cloudflare-workers/serve-static.ts:6` | 5 | **Direct Orphan** | 1.00 |
| `KVAssetOptions` | `type_alias` | `src/adapter/cloudflare-workers/utils.ts:5` | 5 | **Direct Orphan** | 1.00 |
| `FetchEvent` | `interface` | `src/adapter/service-worker/types.ts:6` | 9 | **Direct Orphan** | 1.00 |
| `BuildSearchParamsFn` | `type_alias` | `src/client/types.ts:30` | 1 | **Direct Orphan** | 1.00 |
| `TypedURL` | `interface` | `src/client/types.ts:215` | 16 | **Direct Orphan** | 1.00 |
| `InferRequestOptionsType` | `type_alias` | `src/client/types.ts:273` | 7 | **Direct Orphan** | 1.00 |
| `FilterClientResponseByStatusCode` | `type_alias` | `src/client/types.ts:284` | 9 | **Direct Orphan** | 1.00 |
| `ObjectType` | `type_alias` | `src/client/types.ts:328` | 3 | **Direct Orphan** | 1.00 |
| `Renderer` | `type_alias` | `src/context.ts:78` | 1 | **Direct Orphan** | 1.00 |
| `Layout` | `type_alias` | `src/context.ts:88` | 1 | **Direct Orphan** | 1.00 |
| `TEXT_PLAIN` | `constant` | `src/context.ts:279` | 1 | **Direct Orphan** | 1.00 |
| `Accept` | `interface` | `src/helper/accepts/accepts.ts:5` | 5 | **Direct Orphan** | 1.00 |
| `acceptsConfig` | `interface` | `src/helper/accepts/accepts.ts:11` | 5 | **Direct Orphan** | 1.00 |
| `acceptsOptions` | `interface` | `src/helper/accepts/accepts.ts:17` | 3 | **Direct Orphan** | 1.00 |
| `isPseudoGlobalSelectorRe` | `constant` | `src/helper/css/common.ts:3` | 1 | **Direct Orphan** | 1.00 |
| `EXTERNAL_CLASS_NAMES` | `constant` | `src/helper/css/common.ts:10` | 1 | **Direct Orphan** | 1.00 |
| `IS_CSS_ESCAPED` | `constant` | `src/helper/css/common.ts:21` | 1 | **Direct Orphan** | 1.00 |
| `SSG_CONTEXT` | `constant` | `src/helper/ssg/middleware.ts:5` | 1 | **Direct Orphan** | 1.00 |
| `SSG_DISABLED_RESPONSE` | `constant` | `src/helper/ssg/middleware.ts:13` | 10 | **Direct Orphan** | 1.00 |
| `SSGParams` | `type_alias` | `src/helper/ssg/middleware.ts:27` | 1 | **Direct Orphan** | 1.00 |
| `AddedSSGDataRequest` | `type_alias` | `src/helper/ssg/middleware.ts:36` | 3 | **Direct Orphan** | 1.00 |
| `filterStaticGenerateRoutes` | `function` | `src/helper/ssg/utils.ts:66` | 11 | **Direct Orphan** | 1.00 |
| `getNameSpaceContext` | `function` | `src/jsx/base.ts:56` | 1 | **Transitive Dead** | 1.00 |
| `booleanAttributes` | `constant` | `src/jsx/base.ts:85` | 28 | **Direct Orphan** | 1.00 |
| `FallbackRender` | `type_alias` | `src/jsx/components.ts:44` | 1 | **Direct Orphan** | 1.00 |
| `DOM_ERROR_HANDLER` | `constant` | `src/jsx/constants.ts:2` | 1 | **Direct Orphan** | 1.00 |
| `DOM_INTERNAL_TAG` | `constant` | `src/jsx/constants.ts:4` | 1 | **Direct Orphan** | 1.00 |
| `globalContexts` | `constant` | `src/jsx/context.ts:13` | 1 | **Direct Orphan** | 1.00 |
| `Root` | `interface` | `src/jsx/dom/client.ts:11` | 4 | **Direct Orphan** | 1.00 |
| `RootOptions` | `type_alias` | `src/jsx/dom/client.ts:15` | 1 | **Direct Orphan** | 1.00 |
| `title` | `function` | `src/jsx/dom/intrinsic-element/components.ts:257` | 14 | **Direct Orphan** | 1.00 |
| `script` | `function` | `src/jsx/dom/intrinsic-element/components.ts:272` | 11 | **Direct Orphan** | 1.00 |
| `style` | `function` | `src/jsx/dom/intrinsic-element/components.ts:284` | 13 | **Direct Orphan** | 1.00 |
| `link` | `function` | `src/jsx/dom/intrinsic-element/components.ts:298` | 15 | **Direct Orphan** | 1.00 |
| `meta` | `function` | `src/jsx/dom/intrinsic-element/components.ts:314` | 3 | **Direct Orphan** | 1.00 |
| `form` | `function` | `src/jsx/dom/intrinsic-element/components.ts:319` | 68 | **Direct Orphan** | 1.00 |
| `input` | `function` | `src/jsx/dom/intrinsic-element/components.ts:423` | 2 | **Direct Orphan** | 1.00 |
| `button` | `function` | `src/jsx/dom/intrinsic-element/components.ts:426` | 2 | **Direct Orphan** | 1.00 |
| `HasRenderToDom` | `type_alias` | `src/jsx/dom/render.ts:30` | 1 | **Direct Orphan** | 1.00 |
| `PreserveNodeType` | `type_alias` | `src/jsx/dom/render.ts:37` | 3 | **Direct Orphan** | 1.00 |
| `getNameSpaceContext` | `function` | `src/jsx/dom/render.ts:111` | 1 | **Transitive Dead** | 1.00 |
| `RenderToStringOptions` | `interface` | `src/jsx/dom/server.ts:12` | 3 | **Direct Orphan** | 1.00 |
| `RenderToReadableStreamOptions` | `interface` | `src/jsx/dom/server.ts:37` | 11 | **Direct Orphan** | 1.00 |
| `deDupeKeyMap` | `constant` | `src/jsx/intrinsic-element/common.ts:3` | 7 | **Direct Orphan** | 1.00 |
| `domRenderers` | `constant` | `src/jsx/intrinsic-element/common.ts:11` | 1 | **Direct Orphan** | 1.00 |
| `dataPrecedenceAttr` | `constant` | `src/jsx/intrinsic-element/common.ts:13` | 1 | **Direct Orphan** | 1.00 |
| `isStylesheetLinkWithPrecedence` | `function` | `src/jsx/intrinsic-element/common.ts:15` | 2 | **Transitive Dead** | 1.00 |
| `shouldDeDupeByKey` | `function` | `src/jsx/intrinsic-element/common.ts:18` | 6 | **Transitive Dead** | 1.00 |
| `title` | `function` | `src/jsx/intrinsic-element/components.ts:121` | 15 | **Direct Orphan** | 1.00 |
| `script` | `function` | `src/jsx/intrinsic-element/components.ts:136` | 14 | **Direct Orphan** | 1.00 |
| `style` | `function` | `src/jsx/intrinsic-element/components.ts:151` | 11 | **Direct Orphan** | 1.00 |
| `link` | `function` | `src/jsx/intrinsic-element/components.ts:162` | 9 | **Direct Orphan** | 1.00 |
| `meta` | `function` | `src/jsx/intrinsic-element/components.ts:171` | 7 | **Direct Orphan** | 1.00 |
| `form` | `function` | `src/jsx/intrinsic-element/components.ts:182` | 11 | **Direct Orphan** | 1.00 |
| `input` | `function` | `src/jsx/intrinsic-element/components.ts:207` | 2 | **Direct Orphan** | 1.00 |
| `button` | `function` | `src/jsx/intrinsic-element/components.ts:209` | 2 | **Direct Orphan** | 1.00 |
| `IntrinsicElements` | `interface` | `src/jsx/intrinsic-elements.ts:924` | 1 | **Direct Orphan** | 1.00 |
| `jsxEscape` | `function` | `src/jsx/jsx-runtime.ts:51` | 1 | **Direct Orphan** | 1.00 |
| `StreamingContext` | `constant` | `src/jsx/streaming.ts:30` | 3 | **Direct Orphan** | 1.00 |
| `DEFAULT_OPTIONS` | `constant` | `src/middleware/language/language.ts:51` | 18 | **Direct Orphan** | 1.00 |
| `detectors` | `constant` | `src/middleware/language/language.ts:191` | 6 | **Direct Orphan** | 1.00 |
| `DetectorFunction` | `type_alias` | `src/middleware/language/language.ts:199` | 1 | **Direct Orphan** | 1.00 |
| `Detectors` | `type_alias` | `src/middleware/language/language.ts:202` | 1 | **Direct Orphan** | 1.00 |
| `RequestIdOptions` | `type_alias` | `src/middleware/request-id/request-id.ts:13` | 5 | **Direct Orphan** | 1.00 |
| `PermissionsPolicyDirective` | `type_alias` | `src/middleware/secure-headers/permissions-policy.ts:3` | 4 | **Direct Orphan** | 1.00 |
| `METHOD_NAME_ALL_LOWERCASE` | `constant` | `src/router.ts:13` | 1 | **Direct Orphan** | 1.00 |
| `METHODS` | `constant` | `src/router.ts:17` | 1 | **Direct Orphan** | 1.00 |
| `MESSAGE_MATCHER_IS_ALREADY_BUILT` | `constant` | `src/router.ts:21` | 2 | **Direct Orphan** | 1.00 |
| `Router` | `interface` | `src/router.ts:29` | 24 | **Direct Orphan** | 1.00 |
| `ParamIndexMap` | `type_alias` | `src/router.ts:57` | 1 | **Direct Orphan** | 1.00 |
| `ParamStash` | `type_alias` | `src/router.ts:61` | 1 | **Direct Orphan** | 1.00 |
| `Params` | `type_alias` | `src/router.ts:65` | 1 | **Direct Orphan** | 1.00 |
| `Result` | `type_alias` | `src/router.ts:98` | 1 | **Direct Orphan** | 1.00 |
| `HandlerData` | `type_alias` | `src/router/reg-exp-router/matcher.ts:4` | 1 | **Direct Orphan** | 1.00 |
| `StaticMap` | `type_alias` | `src/router/reg-exp-router/matcher.ts:5` | 1 | **Direct Orphan** | 1.00 |
| `Matcher` | `type_alias` | `src/router/reg-exp-router/matcher.ts:6` | 1 | **Direct Orphan** | 1.00 |
| `MatcherMap` | `type_alias` | `src/router/reg-exp-router/matcher.ts:7` | 1 | **Direct Orphan** | 1.00 |
| `emptyParam` | `constant` | `src/router/reg-exp-router/matcher.ts:9` | 1 | **Direct Orphan** | 1.00 |
| `LABEL_REG_EXP_STR` | `constant` | `src/router/reg-exp-router/node.ts:3` | 1 | **Direct Orphan** | 1.00 |
| `ONLY_WILDCARD_REG_EXP_STR` | `constant` | `src/router/reg-exp-router/node.ts:4` | 1 | **Direct Orphan** | 1.00 |
| `TAIL_WILDCARD_REG_EXP_STR` | `constant` | `src/router/reg-exp-router/node.ts:5` | 1 | **Direct Orphan** | 1.00 |
| `PATH_ERROR` | `constant` | `src/router/reg-exp-router/node.ts:6` | 1 | **Direct Orphan** | 1.00 |
| `ParamAssocArray` | `type_alias` | `src/router/reg-exp-router/node.ts:8` | 1 | **Direct Orphan** | 1.00 |
| `ReplacementMap` | `type_alias` | `src/router/reg-exp-router/trie.ts:5` | 1 | **Direct Orphan** | 1.00 |
| `Bindings` | `type_alias` | `src/types.ts:26` | 1 | **Direct Orphan** | 1.00 |
| `Variables` | `type_alias` | `src/types.ts:27` | 1 | **Direct Orphan** | 1.00 |
| `BlankEnv` | `type_alias` | `src/types.ts:29` | 1 | **Direct Orphan** | 1.00 |
| `ExtractInput` | `type_alias` | `src/types.ts:37` | 5 | **Direct Orphan** | 1.00 |
| `BlankSchema` | `type_alias` | `src/types.ts:48` | 1 | **Direct Orphan** | 1.00 |
| `BlankInput` | `type_alias` | `src/types.ts:49` | 1 | **Direct Orphan** | 1.00 |
| `HTTPResponseError` | `interface` | `src/types.ts:113` | 3 | **Direct Orphan** | 1.00 |
| `HandlerInterface` | `interface` | `src/types.ts:127` | 23 | **Direct Orphan** | 1.00 |
| `Accept` | `interface` | `src/utils/accept.ts:1` | 5 | **Direct Orphan** | 1.00 |
| `Auth` | `type_alias` | `src/utils/basic-auth.ts:7` | 1 | **Direct Orphan** | 1.00 |
| `BodyData` | `type_alias` | `src/utils/body.ts:42` | 3 | **Direct Orphan** | 1.00 |
| `ParseBodyOptions` | `type_alias` | `src/utils/body.ts:46` | 33 | **Direct Orphan** | 1.00 |
| `Pool` | `interface` | `src/utils/concurrent.ts:8` | 3 | **Direct Orphan** | 1.00 |
| `COMPOSED_HANDLER` | `constant` | `src/utils/constants.ts:4` | 1 | **Direct Orphan** | 1.00 |
| `CookieConstraint` | `type_alias` | `src/utils/cookie.ts:31` | 5 | **Direct Orphan** | 1.00 |
| `ResponseHeader` | `type_alias` | `src/utils/headers.ts:266` | 78 | **Direct Orphan** | 1.00 |
| `AcceptHeader` | `type_alias` | `src/utils/headers.ts:345` | 8 | **Direct Orphan** | 1.00 |
| `CustomHeader` | `type_alias` | `src/utils/headers.ts:355` | 1 | **Direct Orphan** | 1.00 |
| `HtmlEscapedCallbackPhase` | `constant` | `src/utils/html.ts:6` | 5 | **Direct Orphan** | 1.00 |
| `StringBuffer` | `type_alias` | `src/utils/html.ts:37` | 1 | **Direct Orphan** | 1.00 |
| `InfoStatusCode` | `type_alias` | `src/utils/http-status.ts:6` | 1 | **Direct Orphan** | 1.00 |
| `SuccessStatusCode` | `type_alias` | `src/utils/http-status.ts:7` | 1 | **Direct Orphan** | 1.00 |
| `DeprecatedStatusCode` | `type_alias` | `src/utils/http-status.ts:8` | 1 | **Direct Orphan** | 1.00 |
| `RedirectStatusCode` | `type_alias` | `src/utils/http-status.ts:9` | 1 | **Direct Orphan** | 1.00 |
| `ClientErrorStatusCode` | `type_alias` | `src/utils/http-status.ts:10` | 30 | **Direct Orphan** | 1.00 |
| `ServerErrorStatusCode` | `type_alias` | `src/utils/http-status.ts:40` | 1 | **Direct Orphan** | 1.00 |
| `UnofficialStatusCode` | `type_alias` | `src/utils/http-status.ts:52` | 1 | **Direct Orphan** | 1.00 |
| `UnOfficalStatusCode` | `type_alias` | `src/utils/http-status.ts:58` | 1 | **Direct Orphan** | 1.00 |
| `ContentlessStatusCode` | `type_alias` | `src/utils/http-status.ts:71` | 1 | **Direct Orphan** | 1.00 |
| `SignatureAlgorithm` | `type_alias` | `src/utils/jwt/jwa.ts:23` | 1 | **Direct Orphan** | 1.00 |
| `SymmetricAlgorithm` | `type_alias` | `src/utils/jwt/jwa.ts:25` | 1 | **Direct Orphan** | 1.00 |
| `AsymmetricAlgorithm` | `type_alias` | `src/utils/jwt/jwa.ts:27` | 11 | **Direct Orphan** | 1.00 |
| `HonoJsonWebKey` | `interface` | `src/utils/jwt/jws.ts:23` | 3 | **Direct Orphan** | 1.00 |
| `SignatureKey` | `type_alias` | `src/utils/jwt/jws.ts:27` | 1 | **Direct Orphan** | 1.00 |
| `TokenHeader` | `interface` | `src/utils/jwt/jwt.ts:38` | 5 | **Direct Orphan** | 1.00 |
| `VerifyOptions` | `type_alias` | `src/utils/jwt/jwt.ts:78` | 12 | **Direct Orphan** | 1.00 |
| `VerifyOptionsWithAlg` | `type_alias` | `src/utils/jwt/jwt.ts:91` | 4 | **Direct Orphan** | 1.00 |
| `CryptoKeyUsage` | `enum` | `src/utils/jwt/types.ts:117` | 10 | **Direct Orphan** | 1.00 |
| `JWTPayload` | `type_alias` | `src/utils/jwt/types.ts:131` | 24 | **Direct Orphan** | 1.00 |
| `utf8Encoder` | `constant` | `src/utils/jwt/utf8.ts:6` | 1 | **Direct Orphan** | 1.00 |
| `utf8Decoder` | `constant` | `src/utils/jwt/utf8.ts:7` | 1 | **Direct Orphan** | 1.00 |
| `BaseMime` | `type_alias` | `src/utils/mime.ts:33` | 1 | **Direct Orphan** | 1.00 |
| `Expect` | `type_alias` | `src/utils/types.ts:7` | 1 | **Direct Orphan** | 1.00 |
| `Equal` | `type_alias` | `src/utils/types.ts:8` | 2 | **Direct Orphan** | 1.00 |
| `NotEqual` | `type_alias` | `src/utils/types.ts:10` | 1 | **Direct Orphan** | 1.00 |
| `RemoveBlankRecord` | `type_alias` | `src/utils/types.ts:18` | 2 | **Direct Orphan** | 1.00 |
| `IfAnyThenEmptyObject` | `type_alias` | `src/utils/types.ts:21` | 1 | **Direct Orphan** | 1.00 |
| `JSONPrimitive` | `type_alias` | `src/utils/types.ts:23` | 1 | **Direct Orphan** | 1.00 |
| `JSONArray` | `type_alias` | `src/utils/types.ts:24` | 1 | **Direct Orphan** | 1.00 |
| `JSONObject` | `type_alias` | `src/utils/types.ts:25` | 3 | **Direct Orphan** | 1.00 |
| `InvalidJSONValue` | `type_alias` | `src/utils/types.ts:28` | 1 | **Direct Orphan** | 1.00 |
| `JSONValue` | `type_alias` | `src/utils/types.ts:39` | 1 | **Direct Orphan** | 1.00 |
| `JSONParsed` | `type_alias` | `src/utils/types.ts:53` | 35 | **Direct Orphan** | 1.00 |
| `Simplify` | `type_alias` | `src/utils/types.ts:93` | 1 | **Direct Orphan** | 1.00 |
| `SimplifyDeepArray` | `type_alias` | `src/utils/types.ts:98` | 3 | **Direct Orphan** | 1.00 |
| `InterfaceToType` | `type_alias` | `src/utils/types.ts:102` | 1 | **Direct Orphan** | 1.00 |
| `RequiredKeysOf` | `type_alias` | `src/utils/types.ts:104` | 6 | **Direct Orphan** | 1.00 |
| `HasRequiredKeys` | `type_alias` | `src/utils/types.ts:111` | 2 | **Direct Orphan** | 1.00 |
| `IsAny` | `type_alias` | `src/utils/types.ts:114` | 1 | **Direct Orphan** | 1.00 |
| `StringLiteralUnion` | `type_alias` | `src/utils/types.ts:120` | 1 | **Direct Orphan** | 1.00 |
| `Pattern` | `type_alias` | `src/utils/url.ts:6` | 1 | **Direct Orphan** | 1.00 |
| `decodeURIComponent_` | `constant` | `src/utils/url.ts:337` | 1 | **Direct Orphan** | 1.00 |
| `IsLiteralUnion` | `type_alias` | `src/validator/utils.ts:9` | 5 | **Direct Orphan** | 1.00 |
| `ExtractValidationResponse` | `type_alias` | `src/validator/validator.ts:28` | 17 | **Direct Orphan** | 1.00 |
| `ExtractValidatorOutput` | `type_alias` | `src/validator/validator.ts:175` | 11 | **Direct Orphan** | 1.00 |

## 4. Neuro-Symbolic Patch Verification with Laya ModernBERT-base (`verify`)

Four distinct patch proposals were evaluated against Hono:

### PATCH-1: Clean Additive Function in src/utils/url.ts
- **Target File:** `src/utils/url.ts`
- **Patch Category:** `Harmless Additive (Neural Approved)`
- **Status Verdict:** `APPROVED`
- **Confidence:** `0.98`
- **Calibrated Risk Score:** `0.5009`
- **End-to-End Latency:** `991.05 ms` (CPU)
- **Subgraph Nodes in Neighborhood:** `2`

```dsl
[DIFF_TARGET] src/utils/url.ts::getCleanUrl, <module> (MODIFIED)
[METADATA] File: src/utils/url.ts | OldLines: [] | NewLines: [338, 339, 340, 341] | Nodes: 2 | Edges: 0
[NODES]
N0: src/utils/url.ts::getCleanUrl [const getCleanUrl = (url: string) => ...] (SEED, MODIFIED)
N1: src/utils/url.ts::<module> [# module src/utils/url.ts] (SEED, MODIFIED)
[EDGES]
(none)
[GATE]
STATUS: APPROVED (conf: 0.98)
CYCLES: 0
VIOLATIONS: NONE
```

### PATCH-2: Malformed TypeScript Syntax Invariant Violation
- **Target File:** `src/utils/url.ts`
- **Patch Category:** `Syntax Violation (AST Guard)`
- **Status Verdict:** `REJECTED`
- **Confidence:** `1.00`
- **Calibrated Risk Score:** `1.0000`
- **End-to-End Latency:** `0.71 ms` (CPU)
- **Subgraph Nodes in Neighborhood:** `0`
- **Violations Detected:**
  - ✖ `SYNTAX_ERROR: SyntaxError at line 1:1: Unexpected TypeScript syntax near 'export const broken = (a: string => {'`

```dsl
[DIFF_TARGET] src/utils/url.ts (SYNTAX_ERROR)
[GATE]
STATUS: REJECTED
VIOLATIONS:
  - SYNTAX_ERROR: SyntaxError at line 1:1: Unexpected TypeScript syntax near 'export const broken = (a: string => {'
```

### PATCH-3: Contract Arity Breaking Change in splitPath
- **Target File:** `src/utils/url.ts`
- **Patch Category:** `Breaking Contract (Arity Mismatch)`
- **Status Verdict:** `REJECTED`
- **Confidence:** `1.00`
- **Calibrated Risk Score:** `0.9500`
- **End-to-End Latency:** `61.02 ms` (CPU)
- **Subgraph Nodes in Neighborhood:** `8`
- **Violations Detected:**
  - ✖ `ARITY_MISMATCH: Caller 'src/utils/url.test.ts::<module>' (line 16) missing required argument 'strict' when calling 'splitPath' (requires at least 2 arguments).`
  - ✖ `ARITY_MISMATCH: Caller 'src/utils/url.test.ts::<module>' (line 19) missing required argument 'strict' when calling 'splitPath' (requires at least 2 arguments).`
  - ✖ `ARITY_MISMATCH: Caller 'src/router/trie-router/node.ts::Node' (line 93) missing required argument 'strict' when calling 'splitPath' (requires at least 2 arguments).`
  - ✖ `ARITY_MISMATCH: Caller 'src/router/trie-router/node.ts::Node.search' (line 93) missing required argument 'strict' when calling 'splitPath' (requires at least 2 arguments).`
  - ✖ `ARITY_MISMATCH: Caller 'src/utils/url.ts::splitRoutingPath' (line 19) missing required argument 'strict' when calling 'splitPath' (requires at least 2 arguments).`

```dsl
[DIFF_TARGET] src/utils/url.ts::splitPath (MODIFIED)
[METADATA] File: src/utils/url.ts | OldLines: [8] | NewLines: [8] | Nodes: 5 | Edges: 7
[NODES]
N0: src/utils/url.ts::splitPath [const splitPath = (path: string, strict: boolean) => ...] (SEED, MODIFIED)
N1: src/utils/url.test.ts::<module> [// module src/utils/url.test.ts]
N2: src/router/trie-router/node.ts::Node [class Node]
N3: src/router/trie-router/node.ts::search [search(method: string, path: string): [[T, Params][]]]
[EDGES]
N1 -> N0 (CALLS)
N2 -> N0 (CALLS)
N3 -> N0 (CALLS)
N2 -> N2 (CALLS)
[GATE]
STATUS: REJECTED (conf: 0.95)
CYCLES: 0
VIOLATIONS:
  - ARITY_MISMATCH: Caller 'src/utils/url.test.ts::<module>' (line 16) missing required argument 'strict' when calling 'splitPath' (requires at least 2 arguments).
  - ARITY_MISMATCH: Caller 'src/utils/url.test.ts::<module>' (line 19) missing required argument 'strict' when calling 'splitPath' (requires at least 2 arguments).
  - ARITY_MISMATCH: Caller 'src/router/trie-router/node.ts::Node' (line 93) missing required argument 'strict' when calling 'splitPath' (requires at least 2 arguments).
  - ... (2 more)
[TRUNCATED: 1 peripheral nodes pruned for token budget]
```

### PATCH-4: Recursive Mutual Call Cycle Injection in TypeScript
- **Target File:** `src/utils/url.ts`
- **Patch Category:** `Topological Cycle (Hard Veto)`
- **Status Verdict:** `REJECTED`
- **Confidence:** `1.00`
- **Calibrated Risk Score:** `0.9500`
- **End-to-End Latency:** `56.63 ms` (CPU)
- **Subgraph Nodes in Neighborhood:** `5`
- **Violations Detected:**
  - ✖ `CIRCULAR_DEPENDENCY: Detected call cycle: pingTS -> pongTS -> pingTS`
- **Cycles Detected:**
  - ↺ `src/utils/url.ts::pingTS -> src/utils/url.ts::pongTS`

```dsl
[DIFF_TARGET] src/utils/url.ts::pingTS, pongTS, <module> (MODIFIED)
[METADATA] File: src/utils/url.ts | OldLines: [] | NewLines: [338..345] (8 lines) | Nodes: 3 | Edges: 2
[NODES]
N0: src/utils/url.ts::pingTS [function pingTS(): void] (SEED, MODIFIED)
N1: src/utils/url.ts::pongTS [function pongTS(): void] (SEED, MODIFIED)
N2: src/utils/url.ts::<module> [# module src/utils/url.ts] (SEED, MODIFIED)
[EDGES]
N0 -> N1 (CALLS)
N1 -> N0 (CALLS)
[GATE]
STATUS: REJECTED (conf: 0.95)
CYCLES: 1
VIOLATIONS:
  - CIRCULAR_DEPENDENCY: Detected call cycle: pingTS -> pongTS -> pingTS
```

---

## 5. Key Takeaways & Architecture Validation

1. **Core Framework Cyclic Imports:** Hono has a 5-module import cycle connecting `compose.ts`, `hono-base.ts`, `types.ts`, `request.ts`, and `context.ts`. Decoupling types into a standalone leaf module would improve bundler tree-shaking and isolation.
2. **Blocking Synchronous Calls in Async Benchmarks (`PERF004`):** Found 9 instances of synchronous Node.js `fs` calls (`mkdirSync`, `writeFileSync`, `existsSync`, `rmSync`) inside `async function` definitions in benchmark and bundle-check scripts.
3. **Nested Loops in JSX DOM & Routers (`PERF001`):** JSX DOM benchmarking code reaches loop depth 4 ($O(N^4)$), and client router parsing reaches depth 2 ($O(N^2)$).
4. **Neuro-Symbolic Gate Accuracy on TypeScript:**
   - Safe additive patch verified by ModernBERT-base 164M in ~890 ms (Risk: `0.5009`).
   - Syntax error blocked in 0.94 ms by AST Parser.
   - Contract arity breaking change across test callers detected in 67.94 ms.
   - Mutual recursive call cycle detected and vetoed in 46.07 ms by Tarjan SCC.