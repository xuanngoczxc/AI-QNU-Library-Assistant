import { useState, useRef, useCallback } from 'react';
import { transcribeVoiceAudio } from '../services/api';

export function useVoiceInput(onTranscribeSuccess: (text: string) => void, onError: (msg: string) => void) {
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const startRecording = useCallback(async () => {
    if (isRecording) return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunksRef.current = [];

      let mimeType = 'audio/webm';
      if (!MediaRecorder.isTypeSupported('audio/webm')) {
        if (MediaRecorder.isTypeSupported('audio/mp4')) mimeType = 'audio/mp4';
        else if (MediaRecorder.isTypeSupported('audio/ogg')) mimeType = 'audio/ogg';
      }

      const recorder = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      recorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });
        stream.getTracks().forEach((t) => t.stop());

        if (audioBlob.size < 1000) {
          onError('Audio quá ngắn, vui lòng thử lại.');
          return;
        }

        try {
          const ext = mimeType.includes('mp4') ? 'mp4' : mimeType.includes('ogg') ? 'ogg' : 'webm';
          const result = await transcribeVoiceAudio(audioBlob, ext);
          if (result.text && result.text.trim()) {
            onTranscribeSuccess(result.text.trim());
          } else {
            onError('Không nhận dạng được giọng nói. Vui lòng thử lại.');
          }
        } catch (err: any) {
          onError(err.message || 'Lỗi nhận dạng giọng nói');
        }
      };

      recorder.start();
      setIsRecording(true);
    } catch (err: any) {
      console.error('Microphone error:', err);
      onError('Không thể truy cập Microphone. Vui lòng cấp quyền trong trình duyệt.');
    }
  }, [isRecording, onTranscribeSuccess, onError]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  }, [isRecording]);

  const toggleRecording = useCallback(() => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  }, [isRecording, startRecording, stopRecording]);

  return {
    isRecording,
    toggleRecording,
    startRecording,
    stopRecording,
  };
}
