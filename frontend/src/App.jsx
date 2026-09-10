import React, { useState, useRef } from 'react';
import {
  UploadCloud,
  FileAudio,
  Play,
  Copy,
  Check,
  Download,
  AlertCircle,
  Zap,
  FolderOpen
} from 'lucide-react';

export default function App() {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [results, setResults] = useState([]);
  const [modelSize, setModelSize] = useState('small');
  const [isProcessing, setIsProcessing] = useState(false);
  const [progressStatus, setProgressStatus] = useState('');
  const [currentIndex, setCurrentIndex] = useState(-1);
  const [error, setError] = useState(null);

  const fileInputRef = useRef(null);
  const folderInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      addFiles(Array.from(e.dataTransfer.files));
    }
  };

  const addFiles = (files) => {
    setError(null);
    const validExtensions = /\.(wav|mp3|m4a|ogg|flac|mp4|aac|webm)$/i;
    const audioFiles = files.filter(f => 
      f.type.startsWith('audio/') || f.type === 'video/mp4' || f.name.match(validExtensions)
    );
    
    if (audioFiles.length < files.length) {
      setError(`Some files were skipped. Only allowed formats: .wav, .mp3, .m4a, .ogg, .flac, .mp4, .aac, .webm`);
    }

    if (audioFiles.length > 0) {
      setSelectedFiles(prev => [...prev, ...audioFiles]);
    }
  };

  const handleTranscribeAll = async () => {
    if (selectedFiles.length === 0) return;
    setIsProcessing(true);
    setError(null);

    let currentResults = [...results];

    for (let i = 0; i < selectedFiles.length; i++) {
      const file = selectedFiles[i];
      setCurrentIndex(i);
      setProgressStatus(`Processing ${i + 1} of ${selectedFiles.length}: ${file.name}...`);

      const formData = new FormData();
      formData.append('file', file);
      formData.append('model_size', modelSize);

      try {
        const response = await fetch('/api/transcribe', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error('Processing failed');
        }

        const data = await response.json();
        data.localAudioUrl = URL.createObjectURL(file);
        
        currentResults = [...currentResults, data];
        setResults(currentResults);
      } catch (err) {
        console.error(err);
        // If error, push a failed result so user knows
        currentResults = [...currentResults, {
          filename: file.name,
          error: true,
          detected_language: 'N/A',
          full_transcription: 'Error processing file.',
          full_translation: 'Error processing file.',
          localAudioUrl: null
        }];
        setResults(currentResults);
      }
    }

    setIsProcessing(false);
    setProgressStatus('');
    setCurrentIndex(-1);
    setSelectedFiles([]); 
  };

  return (
    <div className="h-screen overflow-hidden bg-slate-100 text-slate-800 flex flex-col font-sans selection:bg-hdfc-blue selection:text-white">
      {/* Top Red Brand Strip */}
      <div className="h-1.5 bg-hdfc-red w-full flex-shrink-0"></div>

      {/* Main Corporate Header */}
      <header className="bg-hdfc-dark text-white border-b border-hdfc-darker shadow-sm flex-shrink-0">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-14 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            {/* Official HDFC Bank Logo */}
            <div className="bg-white p-1 rounded-md shadow-xs border border-slate-200 flex items-center justify-center">
              <img
                src="/hdfc-logo.png"
                alt="HDFC Bank Logo"
                className="h-8 w-auto object-contain"
                onError={(e) => {
                  e.target.style.display = 'none';
                }}
              />
            </div>
            <div className="border-l border-slate-600/70 pl-3">
              <div className="flex items-center space-x-2">
                <span className="font-extrabold text-base sm:text-lg tracking-tight text-white">
                  HDFC BANK
                </span>
                <span className="text-slate-400 font-light">|</span>
                <span className="font-semibold text-xs sm:text-sm text-slate-100 tracking-wide">
                  Speach To Text
                </span>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="flex-1 w-full mx-auto px-4 sm:px-6 lg:px-8 py-4 flex flex-col gap-4 overflow-hidden">
        
        {/* Top Controls */}
        <div className="flex flex-col items-center justify-center space-y-3 flex-shrink-0 w-full bg-white p-4 rounded-xl border border-slate-200 shadow-sm max-w-4xl mx-auto">
          
          <div className="flex flex-wrap items-center justify-center gap-6 w-full">
            {/* Drop Zone */}
            <div
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className="flex-1 min-w-[250px] relative rounded-xl border-2 border-dashed border-slate-300 hover:border-hdfc-blue bg-slate-50 hover:bg-blue-50/50 p-4 transition-all duration-200 cursor-pointer flex items-center justify-center text-center h-[90px]"
            >
              <input ref={fileInputRef} type="file" multiple accept="audio/*,.ogg,.wav,.mp3,.m4a,.flac,.mp4,.aac,.webm" className="hidden" onChange={(e) => addFiles(Array.from(e.target.files))} />
              <input ref={folderInputRef} type="file" webkitdirectory="true" directory="true" multiple className="hidden" onChange={(e) => addFiles(Array.from(e.target.files))} />
              
              <div className="flex flex-col items-center space-y-1">
                <UploadCloud className="w-6 h-6 text-hdfc-blue" />
                <span className="text-sm font-bold text-slate-700">Drop files/folders or click</span>
              </div>
            </div>

            <div className="flex flex-col gap-2">
              <button onClick={() => folderInputRef.current?.click()} className="px-4 py-3 bg-slate-100 border border-slate-300 rounded-lg text-xs font-bold text-slate-700 hover:bg-slate-200 flex items-center space-x-2 transition shadow-sm h-full">
                <FolderOpen className="w-5 h-5 text-hdfc-blue" />
                <span>Upload Folder</span>
              </button>
            </div>

            <div className="flex flex-col gap-2 bg-slate-50 p-2.5 rounded-lg border border-slate-200 h-[90px] justify-center">
               <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider px-1">Engine Precision</span>
               <div className="flex items-center space-x-4 text-xs">
                 <label className="flex items-center space-x-1.5 cursor-pointer">
                    <input type="radio" name="model" checked={modelSize === 'small'} onChange={() => setModelSize('small')} className="text-hdfc-blue accent-hdfc-blue" />
                    <span className="font-semibold text-slate-700">Small (Fast)</span>
                 </label>
                 <label className="flex items-center space-x-1.5 cursor-pointer">
                    <input type="radio" name="model" checked={modelSize === 'medium'} onChange={() => setModelSize('medium')} className="text-hdfc-blue accent-hdfc-blue" />
                    <span className="font-semibold text-slate-700">Medium (Accurate)</span>
                 </label>
               </div>
            </div>

            <button
              disabled={selectedFiles.length === 0 || isProcessing}
              onClick={handleTranscribeAll}
              className={`px-6 py-3 rounded-lg font-bold text-sm flex items-center justify-center space-x-2 shadow transition-all h-[90px] min-w-[200px] ${
                selectedFiles.length === 0 || isProcessing
                  ? 'bg-slate-200 text-slate-400 cursor-not-allowed border border-slate-300'
                  : 'bg-hdfc-red hover:bg-hdfc-redHover text-white shadow-red-500/20 active:scale-[0.99]'
              }`}
            >
              {isProcessing ? (
                <>
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  <span>Processing...</span>
                </>
              ) : (
                <>
                  <Zap className="w-5 h-5" />
                  <div className="flex flex-col items-start leading-tight">
                    <span>Analyze & Transcribe</span>
                    <span className="text-[10px] font-medium opacity-90">{selectedFiles.length} item(s) in queue</span>
                  </div>
                </>
              )}
            </button>
          </div>

          {isProcessing && (
            <div className="text-xs text-hdfc-blue font-bold animate-pulse pt-2">{progressStatus}</div>
          )}

          {error && (
            <div className="flex items-center space-x-1.5 text-xs text-red-600 font-semibold bg-red-50 p-2 border border-red-100 rounded mt-2 w-full justify-center">
              <AlertCircle className="w-4 h-4" />
              <span>{error}</span>
            </div>
          )}
        </div>

        {/* Sheet Format Table */}
        <div className="flex-1 bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden flex flex-col min-h-0">
          <div className="overflow-x-auto overflow-y-auto h-full w-full">
            <table className="w-full text-left border-collapse text-sm min-w-[1200px]">
              <thead className="bg-slate-100 text-slate-600 sticky top-0 z-10 shadow-sm border-b border-slate-200">
                <tr>
                  <th className="p-3 font-bold uppercase tracking-wider text-[10px] w-12 text-center bg-slate-100">S.No</th>
                  <th className="p-3 font-bold uppercase tracking-wider text-[10px] w-20 text-center bg-slate-100">Status</th>
                  <th className="p-3 font-bold uppercase tracking-wider text-[10px] w-64 bg-slate-100">Audio Player</th>
                  <th className="p-3 font-bold uppercase tracking-wider text-[10px] w-48 bg-slate-100">Audio Name</th>
                  <th className="p-3 font-bold uppercase tracking-wider text-[10px] w-[25%] bg-slate-100">Original Language (Native)</th>
                  <th className="p-3 font-bold uppercase tracking-wider text-[10px] w-[25%] bg-slate-100">English Translated Content</th>
                  <th className="p-3 font-bold uppercase tracking-wider text-[10px] w-32 text-center bg-slate-100">Detected Lang</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 align-top">
                {results.map((res, idx) => (
                  <tr key={idx} className="hover:bg-blue-50/30 transition-colors group">
                    <td className="p-4 text-slate-500 font-mono text-center align-middle bg-slate-50/50">{idx + 1}</td>
                    <td className="p-4 text-center align-middle">
                      {res.error ? (
                        <span className="text-red-500 font-bold text-xs">Failed</span>
                      ) : (
                        <span className="text-emerald-600 font-bold text-xs flex items-center justify-center space-x-1">
                          <Check className="w-3 h-3"/><span>Done</span>
                        </span>
                      )}
                    </td>
                    <td className="p-4 align-middle">
                      {res.localAudioUrl ? (
                        <audio src={res.localAudioUrl} controls className="w-full h-9 accent-hdfc-blue" />
                      ) : (
                        <span className="text-xs text-slate-400">N/A</span>
                      )}
                    </td>
                    <td className="p-4 font-semibold text-slate-800 break-all align-middle bg-slate-50/30">
                      {res.filename || 'Audio'}
                    </td>
                    <td className="p-4">
                      <div className="text-slate-700 leading-relaxed max-h-48 overflow-y-auto block pr-2 custom-scrollbar">
                        {res.full_transcription || '-'}
                      </div>
                    </td>
                    <td className="p-4">
                      <div className="text-slate-700 leading-relaxed max-h-48 overflow-y-auto block pr-2 custom-scrollbar border-l border-slate-100 pl-4">
                        {res.full_translation || '-'}
                      </div>
                    </td>
                    <td className="p-4 text-center align-middle bg-slate-50/50">
                      {res.error ? '-' : (
                        <div className="flex flex-col items-center">
                          <span className="font-bold text-hdfc-blue uppercase">{res.detected_language}</span>
                          <span className="text-[10px] text-slate-500 font-medium mt-0.5">{((res.language_confidence || 0) * 100).toFixed(0)}% Conf</span>
                        </div>
                      )}
                    </td>
                  </tr>
                ))}
                {results.length === 0 && (
                  <tr>
                    <td colSpan="7" className="p-16 text-center">
                      <div className="flex flex-col items-center justify-center space-y-3 text-slate-400">
                        <FileAudio className="w-12 h-12 text-slate-300" />
                        <span className="text-sm font-medium">No transcripts yet. Upload files or folders to begin processing.</span>
                      </div>
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </main>
    </div>
  );
}
