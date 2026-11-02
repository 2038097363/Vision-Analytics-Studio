import type { CloudSession } from "../types";

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, "")
  ?? "http://127.0.0.1:8765/v1";

interface ProblemDetails {
  code?: string;
  title?: string;
}

const errorMessages: Record<string, string> = {
  invalid_credentials: "邮箱或密码不正确，或邮箱尚未验证",
  rate_limited: "尝试次数过多，请稍后再试",
  invalid_session: "登录状态已失效，请重新登录",
  session_reuse_detected: "检测到旧会话被重复使用，已保护性退出",
};

async function api<T>(path: string, init: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init.headers ?? {}) },
    });
  } catch {
    throw new Error("无法连接账户服务，请确认本机开发服务或正式服务器可用");
  }
  if (!response.ok) {
    const problem = await response.json().catch(() => ({})) as ProblemDetails;
    throw new Error(errorMessages[problem.code ?? ""] ?? problem.title ?? "账户请求失败");
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export async function registerAccount(email: string, displayName: string, password: string): Promise<void> {
  await api<{ status: string }>("/auth/register", {
    method: "POST",
    body: JSON.stringify({ email, display_name: displayName, password }),
  });
}

export async function loginAccount(email: string, password: string): Promise<CloudSession> {
  return api<CloudSession>("/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password, device_name: "Vision Analytics Studio · Windows" }),
  });
}

export async function logoutAccount(session: CloudSession): Promise<void> {
  await api<void>("/auth/logout", {
    method: "POST",
    headers: { Authorization: `Bearer ${session.access_token}` },
  });
}
