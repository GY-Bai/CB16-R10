# DSH 沙箱使用说明书（OCI · bwrap workspace-write）

> 适用环境：日本 OCI 上 `cb16-dsh-web`（`dsh.bayesdesk.com`）派生的会话，权限预设 `workspace-write`，沙箱后端 `bwrap 0.6.3`。
> 事实来源：`@deepseek-ai/dsh-sandbox` 源码（`writableRoots()` / `bwrapProfileArgs()`）+ 2026-09-14 在 OCI 的实测。
> 本文档放在只读区，沙箱内可直接 `cat` 阅读。

## 0. 一句话总结

沙箱**只限制"往哪写文件"**：工作区 + `/tmp` 可写，其余一律只读；**网络不隔离**。
因此 `pip` / `uv` / `curl` / `git` 都能正常跑，但"安装目标"和"缓存目录"必须落在可写区。

---

## 1. 可写边界（源码事实，别猜）

`dsh-sandbox` 对 `workspace-write` 给出的可写根**恰好三个**：

```js
function writableRoots(policy) {
  if (policy.mode !== "workspace-write") return [];
  return [...new Set([
    policy.workspaceRoot,   // 当前会话的工作区
    "/tmp",
    tmpdir()                // OS 临时目录（Linux 即 /tmp）
  ].map(canonicalPath))];
}
```

`dsh-sandbox-policy` 的配置 schema 只有两个键（**没有**"额外可写根"选项）：

```js
Config = z.object({
  mode: "read-only" | "workspace-write" | "danger-full-access",
  workspaceRoot: string
})
```

Linux 上实际执行的 profile（bwrap）：

```
bwrap --ro-bind / /          # 整个根文件系统 → 只读
      --dev /dev --proc /proc
      --die-with-parent
      --tmpfs /tmp           # /tmp 换成新的 tmpfs（可写）
      --bind <workspaceRoot> <workspaceRoot>   # 只把工作区开成可写
      -- <command>
```

**推论**：`~`（家目录）、`~/.cache`、`~/.local`、`~/.config`、`/usr`、`/etc`、`/var`、`/opt` …… 全是只读；只有工作区与 `/tmp` 能写。

---

## 2. 读写一览表

| 位置 | 沙箱内可写？ | 说明 |
|---|---|---|
| `<workspaceRoot>/...`（当前项目目录） | ✅ | 唯一持久可写区 |
| `/tmp/...` | ✅ | tmpfs；**bwrap 进程结束即消失**（同一持久 bash 会话内保留） |
| `~`（`/home/bgy`） | ❌ | `Read-only file system` |
| `~/.cache`、`~/.local`、`~/.config` | ❌ | pip/uv 的默认缓存与安装位置就在这，踩坑重灾区 |
| `/usr`、`/etc`、`/var`、`/opt` | ❌ | 系统目录 |
| 其它会话/项目目录 | ❌ | 除非它正好是你的 workspaceRoot |
| 网络（HTTP/HTTPS） | ✅ | **不隔离**，`curl https://pypi.org` 实测 200 |

> 写失败的标准签名：`Read-only file system`（errno 30）。

---

## 3. pip 实测行为

| 命令 | 结果 |
|---|---|
| `pip install <pkg>`（默认） | ❌ `Read-only file system: ~/.local/lib/python3.9/site-packages` |
| `pip install --user <pkg>` | ❌ 同上 |
| `pip install --target ./vendor <pkg>` | ✅ 装进工作区；用 `PYTHONPATH=./vendor python3 ...` 引用 |
| `python3 -m venv .venv && .venv/bin/pip install <pkg>` | ✅ 推荐 |
| `pip download -d ./dl <pkg>` | ✅ 只下载不安装 |
| `pip install --upgrade pip` | ❌ 要写 `~/.local`，装不了（去 venv 里升级） |

环境自带：Python **3.9.25**、pip **21.3.1**（在 `~/.local`），常用库（numpy/pandas/requests/httpx/bs4/tqdm/huggingface_hub/psutil/cryptography/yt-dlp/yfinance 等）已可用。

**缓存告警**：pip 会提示 `~/.cache/pip` 不可写（仅 WARNING，功能正常）。消除办法二选一：

```bash
PIP_CACHE_DIR="$PWD/.pip-cache" pip install ...   # 持久到项目里
pip install --no-cache-dir ...                    # 或干脆不用缓存
```

---

## 4. uv 实测行为

环境自带 uv **0.12.13**（`~/.local/bin/uv`）。

| 命令 | 结果 |
|---|---|
| `uv init` / `uv add` / `uv run` / `uv sync` | ✅（只要缓存/venv 在工作区） |
| `uv pip install --system <pkg>` | ❌ 写 `/usr/local/lib/...` 只读 |
| `uv` 用默认缓存 `~/.cache/uv` | ❌ `Could not acquire lock ... Read-only file system` |
| `UV_CACHE_DIR=/tmp/uv-cache uv ...` | ✅ 会话内有效（tmpfs） |
| `UV_CACHE_DIR="$PWD/.uv-cache" uv ...` | ✅ **推荐**，持久到项目 |
| `UV_PYTHON_INSTALL_DIR="$PWD/.uv-python" uv python install 3.12` | ✅ 能把托管 CPython 装进工作区（实测 3.12.14 可用） |

**uv 必须设 `UV_CACHE_DIR`**（否则直接报错，不像 pip 只是 warning）。想用托管的 Python，还要同时设 `UV_PYTHON_INSTALL_DIR`（因为默认写 `~/.local/share/uv`，只读）。

推荐姿势：

```bash
export UV_CACHE_DIR="$PWD/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$PWD/.uv-python"   # 需要独立 Python 版本时
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python <pkg>
```

> `uv python install` 可能会警告 "Failed to create Python executable link ... `~/.local/bin` Read-only" —— 这是**无害**的（venv 仍可用，解释器在工作区里）。

---

## 5. 缓存策略（按持久性选）

| 策略 | 持久性 | 适用 |
|---|---|---|
| `UV_CACHE_DIR=/tmp/uv-cache`、`PIP_CACHE_DIR=/tmp/pip-cache` | 会话内（tmpfs） | 临时跑一下就够 |
| `UV_CACHE_DIR="$PWD/.uv-cache"`、`PIP_CACHE_DIR="$PWD/.pip-cache"` | **持久**（跟项目走） | 长期项目；记得写进 `.gitignore` |
| 切到 `danger-full-access` 会话 | 宿主级 `~/.cache` 正常 | 需要共享宿主缓存时；代价是**没有沙箱** |

没有"沙箱 + 共享宿主缓存"的配置项；要那个组合只能改 composition / 自定义 sandbox provider（属上游决策，不建议为省缓存动它）。

---

## 6. `/tmp` 语义（容易踩）

- 是 bwrap 挂进来的**新 tmpfs**，不是宿主的 `/tmp`；
- 对**每一次 confide 调用**是独立的；对持久 bash 工具（同一会话）在会话存活期间保留；
- 会话结束/进程退出 → 内容消失；
- 所以：venv、下载的大包、数据集**别放 `/tmp` 想长期用**，放工作区。

---

## 7. 失败信号速查

| 报错 | 含义 | 处理 |
|---|---|---|
| `Read-only file system` | 写到工作区/`/tmp` 之外 | 换可写路径 |
| `Could not acquire lock ... ~/.cache/uv` | uv 默认缓存在只读区 | 设 `UV_CACHE_DIR` |
| `Defaulting to user installation because normal site-packages is not writeable` | pip 想装 `~/.local` | 用 `--target` 或 venv |
| `Permission denied`（无 `Read-only`） | 可能是 SELinux/权限 | 换路径或看 §9 |

---

## 8. 自检命令（复制即用）

```bash
# 我是谁、在哪
id; pwd
# 哪些地方能写
for p in "$PWD" /tmp "$HOME" "$HOME/.cache" /usr; do
  printf '%-20s ' "$p"; (echo probe > "$p/.dsh_sandbox_probe" 2>/dev/null && rm -f "$p/.dsh_sandbox_probe" && echo WRITABLE) || echo READ-ONLY
done
# 网络是否通
curl -sS -m 10 -o /dev/null -w 'pypi=%{http_code}\n' https://pypi.org/simple/
# 沙箱探针（等价于 DSH 自身的探测）
bwrap --ro-bind / / --dev /dev --proc /proc --die-with-parent -- true; echo "bwrap=$?"
```

---

## 9. 和山西 runner 的区别（别混淆）

| 现象 | 性质 |
|---|---|
| OCI 沙箱里 `~/.cache/uv`、`~/.cache/pip` 只读 | ✅ **设计**：`workspace-write` 只承诺"工作区 + 临时目录"可写，防止 agent 往 `~/.local/bin`、`~/.bashrc` 等持久化 |
| 山西 runner 容器 `/cb16/uv-cache` 为 `root:root 0755`，而 `UV_CACHE_DIR=/cb16/uv-cache` | ❌ **真 infra 缺陷**（runner uid 1001 写不进去）；见 `docs/infra/SHANXI_DOCKER_RUNNER_CONTRACT.md` §9 缺口 2 |
| 山西 `/cb16/worker` 同理（`root:root 0755`，却挂着 `CB16_CI_WORKER_ROOT`） | ❌ 真缺陷，同 §9 缺口 3 |

一句话：**沙箱的"缓存不可写"是边界设计；runner 的"缓存不可写"是配置错误。**

---

## 10. 想改边界怎么办

1. 只是要装包/缓存 → 用本文 §3–§5 的工作区内方案，不要改沙箱；
2. 需要跨会话共享缓存 → 把缓存目录做成工作区内的固定路径（或项目模板里预置 `.uv-cache`）；
3. 真要放宽边界（例如允许写某个共享目录）→ 当前 schema 不支持；需要改 `dsh-sandbox-local` 的 profile 构造或自写 provider，并走 Sol/Astra 审核，不在会话里临时改；
4. 需要宿主级工具链 → 单独开 `danger-full-access` 会话，明确知道此时**没有沙箱**。

---

*生成日期：2026-09-14 · 依据版本：dsh 0.1.0-rc.6 / bwrap 0.6.3 / uv 0.12.13 / Python 3.9.25 + pip 21.3.1*
