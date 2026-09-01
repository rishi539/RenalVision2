import React, { useState, useRef } from 'react';
import { UploadCloud, Image as ImageIcon, Sparkles, RefreshCw } from 'lucide-react';

const ImageUploader = ({ onAnalyze, loading }) => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [base64Image, setBase64Image] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileChange = (file) => {
    if (!file || !file.type.startsWith('image/')) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      setSelectedImage(e.target.result);
      // Remove header prefix for base64 string
      const rawB64 = e.target.result.split(',')[1];
      setBase64Image(rawB64);
    };
    reader.readAsDataURL(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleAnalyzeClick = () => {
    if (base64Image) {
      onAnalyze(base64Image);
    }
  };

  return (
    <div className="glass-card">
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px' }}>
        <div style={{ width: '32px', height: '32px', borderRadius: '8px', background: 'rgba(59, 130, 246, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <ImageIcon size={18} color="var(--accent-blue)" />
        </div>
        <h3 style={{ fontSize: '16px', fontWeight: '700' }}>CT Scan Input</h3>
      </div>

      <div
        className={`upload-zone ${selectedImage ? 'has-image' : ''} ${isDragOver ? 'drag-over' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
        onClick={() => !selectedImage && fileInputRef.current?.click()}
      >
        <input
          type="file"
          ref={fileInputRef}
          style={{ display: 'none' }}
          accept="image/*"
          onChange={(e) => e.target.files?.[0] && handleFileChange(e.target.files[0])}
        />

        {selectedImage ? (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '14px' }}>
            <img
              src={selectedImage}
              alt="CT Scan Preview"
              style={{ maxHeight: '220px', maxWidth: '100%', borderRadius: '10px', boxShadow: '0 4px 20px rgba(0,0,0,0.3)' }}
            />
            <button
              className="logout-btn"
              onClick={(e) => {
                e.stopPropagation();
                setSelectedImage(null);
                setBase64Image(null);
              }}
            >
              <RefreshCw size={12} style={{ display: 'inline', marginRight: '4px' }} /> Change Image
            </button>
          </div>
        ) : (
          <div>
            <UploadCloud size={44} color="var(--accent-blue)" style={{ marginBottom: '12px', opacity: 0.8 }} />
            <h4 style={{ fontSize: '14px', fontWeight: '600', marginBottom: '4px' }}>Drop your CT scan here</h4>
            <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Supports PNG, JPG, JPEG, WEBP or DICOM renders</p>
          </div>
        )}
      </div>

      <button
        className="predict-btn"
        disabled={!base64Image || loading}
        onClick={handleAnalyzeClick}
      >
        {loading ? (
          <span>Analyzing CT Scan & Generating Grad-CAM...</span>
        ) : (
          <>
            <Sparkles size={18} />
            <span>Analyze Image</span>
          </>
        )}
      </button>
    </div>
  );
};

export default ImageUploader;
