export interface BookSource {
  title: string;
  author?: string;
  year?: string;
  subject?: string;
  link?: string;
  source?: string;
  doc_type?: string;
  format?: string;
  major?: string;
  publisher?: string;
  place?: string;
  edition?: string;
  bilingual_title?: string;
  course?: string;
  keywords?: string[];
  location?: string;
  abstract?: string;
  ddc?: string;
  notes?: string;
  description?: string;
  is_pdf?: boolean;
  pdf_id?: string;
  file_name?: string;
  display_name?: string;
  score?: number;
}

export interface FollowUpSuggestion {
  label: string;
  query: string;
}

export interface ChatMessage {
  id: string;
  sender: 'user' | 'bot';
  text: string;
  timestamp: string;
  isStreaming?: boolean;
  sources?: BookSource[];
  totalFound?: number;
  followUps?: FollowUpSuggestion[];
  isError?: boolean;
}

export interface MajorItem {
  nganh: string;
  count: number;
}

export interface FacultyItem {
  khoa: string;
  total: number;
  nganh: MajorItem[];
}

export interface FacultiesResponse {
  faculties: FacultyItem[];
  total_faculties: number;
  total_majors?: number;
}

export interface BrowseDocumentResponse {
  results: BookSource[];
  total: number;
  page: number;
  limit: number;
  error?: string;
}

export interface AdminFileItem {
  name: string;
  size: number;
  mtime: string;
}

export interface AdminDashboardData {
  documents?: {
    total?: number;
  };
  updated?: {
    last_data_update?: string;
  };
  files?: AdminFileItem[];
}

export interface DailyQueryItem {
  date: string;
  queries: number;
  no_result?: number;
}

export interface IntentDistItem {
  intent: string;
  count: number;
  pct: number;
}

export interface TopQueryItem {
  query: string;
  count: number;
}

export interface AnalyticsResponse {
  total: number;
  unique_days?: number;
  daily_queries?: DailyQueryItem[];
  daily_no_result?: { date: string; count: number }[];
  intent_distribution?: IntentDistItem[];
  top_queries?: TopQueryItem[];
}

export interface RecentActivityEntry {
  query: string;
  result_count?: number;
  latency_ms?: number;
  ts: string;
  error?: boolean;
}

export interface RecentActivityResponse {
  entries: RecentActivityEntry[];
}

