import { create } from 'zustand';

const useDocumentStore = create((set, get) => ({
  documentStats: null,
  isLoading: false,
  guestId: null,
  
  setGuestId: (id) => set({ guestId: id }),
  
  setLoading: (loading) => set({ isLoading: loading }),
  
  setDocumentStats: (stats) => set({ documentStats: stats }),
  
  fetchDocumentStats: async () => {
    const { guestId } = get();
    if (!guestId) return;
    
    set({ isLoading: true });
    try {
      const response = await fetch(`http://localhost:8000/documents/${guestId}`);
      if (response.ok) {
        const stats = await response.json();
        set({ documentStats: stats });
      } else {
        console.error('Failed to fetch document stats');
      }
    } catch (error) {
      console.error('Error fetching document stats:', error);
    } finally {
      set({ isLoading: false });
    }
  },
  
  deleteDocument: async (documentName) => {
    const { guestId, fetchDocumentStats } = get();
    if (!guestId) return;
    
    try {
      const response = await fetch(`http://localhost:8000/documents/${guestId}/${documentName}`, {
        method: 'DELETE',
      });
      
      if (response.ok) {
        // Refresh document stats after deletion
        await fetchDocumentStats();
        return { success: true };
      } else {
        const errorData = await response.json();
        return { success: false, error: errorData.detail || 'Failed to delete document' };
      }
    } catch (error) {
      console.error('Error deleting document:', error);
      return { success: false, error: 'Network error occurred' };
    }
  }
}));

export default useDocumentStore;