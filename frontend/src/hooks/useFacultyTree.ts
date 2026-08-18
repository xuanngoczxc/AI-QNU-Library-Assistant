import { useState, useEffect, useCallback } from 'react';
import type { FacultyItem } from '../types';
import { fetchFaculties } from '../services/api';

export function useFacultyTree() {
  const [faculties, setFaculties] = useState<FacultyItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedMajor, setSelectedMajor] = useState<string | null>(null);
  const [selectedKhoa, setSelectedKhoa] = useState<string | null>(null);
  const [openGroups, setOpenGroups] = useState<Record<string, boolean>>({});

  useEffect(() => {
    let isMounted = true;
    fetchFaculties()
      .then((data) => {
        if (isMounted) {
          const list = data.faculties || [];
          setFaculties(list.slice().sort((a, b) => (b.total || 0) - (a.total || 0)));
          setLoading(false);
        }
      })
      .catch((err) => {
        if (isMounted) {
          setError(err.message);
          setLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, []);

  const toggleGroup = useCallback((khoaName: string) => {
    setOpenGroups((prev) => ({
      ...prev,
      [khoaName]: !prev[khoaName],
    }));
  }, []);

  const selectMajor = useCallback((nganh: string, khoa: string) => {
    setSelectedMajor(nganh);
    setSelectedKhoa(khoa);
    // Tự động mở khoa chứa ngành này
    setOpenGroups((prev) => ({
      ...prev,
      [khoa]: true,
    }));
  }, []);

  const clearMajor = useCallback(() => {
    setSelectedMajor(null);
    setSelectedKhoa(null);
  }, []);

  return {
    faculties,
    loading,
    error,
    selectedMajor,
    selectedKhoa,
    openGroups,
    toggleGroup,
    selectMajor,
    clearMajor,
  };
}
