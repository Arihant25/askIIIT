// API service for admin operations
export class AdminAPIService {
  private static baseURL = '';

  private static getAuthHeaders() {
    // Since auth_token is httpOnly, we can't access it from JavaScript
    // The Next.js API routes will handle forwarding cookies to the backend
    return {};
  }

  static async getSystemInfo() {
    const response = await fetch('/api/admin/system-info', {
      credentials: 'include',
    });

    if (!response.ok) {
      throw new Error('Failed to fetch system info');
    }

    return response.json();
  }

  static async getDocuments() {
    const response = await fetch('/api/documents', {
      credentials: 'include',
    });

    if (!response.ok) {
      throw new Error('Failed to fetch documents');
    }

    return response.json();
  }

  static async uploadDocument(formData: FormData) {
    const response = await fetch('/api/documents', {
      method: 'POST',
      body: formData,
      credentials: 'include',
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Upload failed');
    }

    return response.json();
  }

  static async deleteDocument(docId: string) {
    const response = await fetch(`/api/admin/documents/${docId}`, {
      method: 'DELETE',
      credentials: 'include',
    });

    if (!response.ok) {
      throw new Error('Failed to delete document');
    }

    return response.json();
  }

  static async reindexDocuments() {
    const response = await fetch('/api/admin/reindex', {
      method: 'POST',
      credentials: 'include',
    });

    if (!response.ok) {
      throw new Error('Reindexing failed');
    }

    return response.json();
  }

  static async getUserInfo() {
    const response = await fetch('/api/auth/user', {
      credentials: 'include',
    });

    if (!response.ok) {
      throw new Error('Failed to get user info');
    }

    return response.json();
  }

  static async getLoginURL() {
    const response = await fetch('/api/auth/login');

    if (!response.ok) {
      throw new Error('Failed to get login URL');
    }

    return response.json();
  }
}
