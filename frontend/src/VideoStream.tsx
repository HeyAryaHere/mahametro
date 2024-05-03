import React, { useEffect, useRef } from 'react';
import axios from 'axios';

const VideoStream: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const fetchVideoStream = async () => {
      try {
        const response = await axios.get('/video_feed', {
          responseType: 'arraybuffer',
          timeout: 0,
        });

        const blob = new Blob([response.data], { type: 'image/jpeg' });
        const url = URL.createObjectURL(blob);

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
        URL.revokeObjectURL(videoRef.current.src);
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
