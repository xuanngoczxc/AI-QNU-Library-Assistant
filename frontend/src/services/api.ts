import type {
  FacultiesResponse,
  BrowseDocumentResponse,
  BookSource,
  FollowUpSuggestion
} from '../types';

export interface StreamChatCallbacks {
  onChunk: (chunk: string) => void;
  onSources?: (sources: BookSource[]) => void;
  onTotalFound?: (total: number) => void;
  onFollowUps?: (followUps: FollowUpSuggestion[]) => void;
  onDone?: () => void;
  onError?: (errorMsg: string) => void;
}

export async function fetchChatStream(
  query: string,
  sessionId: string,
  signal: AbortSignal,
  callbacks: StreamChatCallbacks
): Promise<void> {
  const response = await fetch('/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      session_id: sessionId,
      client_ip: '',
      user_agent: navigator.userAgent,
    }),
    signal,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${response.status}`);
  }

  const contentType = response.headers.get('content-type') || '';

  if (contentType.includes('text/event-stream')) {
    if (!response.body) throw new Error('Response body is null');
    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const jsonStr = line.slice(6).trim();
        if (!jsonStr) continue;

        try {
          const payload = JSON.parse(jsonStr);
          if (payload.type === 'chunk') {
            callbacks.onChunk(payload.text || '');
          } else if (payload.type === 'sources') {
            callbacks.onSources?.(payload.sources || []);
          } else if (payload.type === 'total_found') {
            callbacks.onTotalFound?.(payload.total_found || 0);
          } else if (payload.type === 'follow_ups') {
            callbacks.onFollowUps?.(payload.follow_up_suggestions || []);
          } else if (payload.type === 'done') {
            callbacks.onDone?.();
          } else if (payload.type === 'error') {
            callbacks.onError?.(payload.message || 'Lỗi xử lý phản hồi');
          }
        } catch {
          // Ignore json parse error for partial lines
        }
      }
    }
  } else {
    // Non-streaming JSON fallback
    const data = await response.json();
    callbacks.onChunk(data.answer || 'Xin lỗi, không có phản hồi.');
    callbacks.onSources?.(data.sources || []);
    callbacks.onTotalFound?.(data.total_found || data.sources?.length || 0);
    callbacks.onFollowUps?.(data.follow_up_suggestions || []);
    callbacks.onDone?.();
  }
}

export async function fetchFaculties(): Promise<FacultiesResponse> {
  const res = await fetch('/api/faculties');
  if (!res.ok) throw new Error('Không thể tải danh sách khoa');
  return res.json();
}

export async function fetchBrowseDocuments(page: number, limit = 20): Promise<BrowseDocumentResponse> {
  const res = await fetch(`/documents?page=${page}&limit=${limit}`);
  if (!res.ok) throw new Error('Không thể tải danh mục tài liệu');
  return res.json();
}

export async function searchDocuments(query: string, sessionId: string): Promise<{ sources: BookSource[] }> {
  const res = await fetch('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, session_id: sessionId }),
  });
  if (!res.ok) throw new Error('Tìm kiếm thất bại');
  return res.json();
}

export async function transcribeVoiceAudio(audioBlob: Blob, ext = 'webm'): Promise<{ text: string; language?: string }> {
  const formData = new FormData();
  formData.append('audio', audioBlob, `voice.${ext}`);
  formData.append('language', 'vi');

  const res = await fetch('/api/voice/transcribe', {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Lỗi nhận dạng: HTTP ${res.status}`);
  }

  return res.json();
}

export async function loginAdmin(password: string): Promise<{ token: string; status: string }> {
  const res = await fetch('/api/admin/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ password }),
  });

  if (!res.ok) {
    throw new Error('Sai mật khẩu!');
  }

  return res.json();
}

export async function fetchAdminDashboard(token: string) {
  const res = await fetch('/api/admin/dashboard', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ token }),
  });
  if (!res.ok) throw new Error('Không thể tải thông tin admin');
  return res.json();
}

export async function uploadAdminData(token: string, file: File) {
  const fd = new FormData();
  fd.append('token', token);
  fd.append('file', file);
  const res = await fetch('/api/admin/upload-data', {
    method: 'POST',
    body: fd,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || 'Upload thất bại');
  return data;
}

export async function deleteAdminFile(token: string, fileName: string) {
  const res = await fetch('/api/admin/delete-file', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ token, file_name: fileName }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || 'Xóa thất bại');
  return data;
}

export async function fetchUsageAnalytics(dateFrom?: string, dateTo?: string) {
  let url = '/api/usage/analytics';
  const params: string[] = [];
  if (dateFrom) params.push(`date_from=${dateFrom}`);
  if (dateTo) params.push(`date_to=${dateTo}`);
  if (params.length) url += `?${params.join('&')}`;

  const res = await fetch(url);
  if (!res.ok) throw new Error('Không thể tải dữ liệu phân tích');
  return res.json();
}

export async function fetchRecentActivity(limit = 10) {
  const res = await fetch(`/api/usage/recent?n=${limit}`);
  if (!res.ok) throw new Error('Không thể tải hoạt động gần đây');
  return res.json();
}

