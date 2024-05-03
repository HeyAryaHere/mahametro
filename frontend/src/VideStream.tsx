import React, { useEffect, useRef } from 'react';
import axios, { AxiosResponse } from 'axios';

const VideoStream: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const fetchVideoStream = async () => {
      try {
        const response: AxiosResponse<ArrayBuffer> = await axios.get('http://127.0.0.1:5000/video_feed', {
          responseType: 'arraybuffer',
          headers: {
            'Content-Type': 'multipart/x-mixed-replace',
          },
          timeout: 0, // Disable Axios timeout
        });

        const blob = new Blob([response.data], { type: 'image/jpeg' });
        const url = window.URL.createObjectURL(blob);
        if (videoRef.current) {
          videoRef.current.src = url;
        }
      } catch (error) {
        console.error('Error fetching video stream:', error);
      }
    };

    fetchVideoStream();

    return () => {
      if (videoRef.current && videoRef.current.src) {
        window.URL.revokeObjectURL(videoRef.current.src);
      }
    };
  }, []);

  return (
    <div>
      <h2>Live Video Stream</h2>
      <video ref={videoRef} controls autoPlay />
    </div>
  );
};

export default VideoStream;
