import { useState } from "react";
import { Upload, File, X } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface UploadCardProps {
  onUpload?: (file: File) => void;
}

export default function UploadCard({ onUpload }: UploadCardProps) {
  const [file, setFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFileSelect = (selectedFile: File) => {
    setFile(selectedFile);
    if (onUpload) {
      onUpload(selectedFile);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0]);
    }
  };

  const removeFile = () => {
    setFile(null);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm
                 dark:border-gray-800 dark:bg-white/[0.03]"
    >
      <h3 className="text-lg font-semibold text-gray-800 dark:text-white/90 mb-4">
        Upload File
      </h3>

      <AnimatePresence mode="wait">
        {!file ? (
          <motion.label
            key="upload"
            htmlFor="file-upload"
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.95, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className={`flex flex-col items-center justify-center w-full h-40 border-2 border-dashed rounded-xl cursor-pointer
              transition-colors duration-300
              ${
                dragOver
                  ? "border-indigo-500 bg-indigo-50/60 dark:bg-indigo-500/10"
                  : "border-gray-300 dark:border-gray-700"
              }`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                handleFileSelect(e.dataTransfer.files[0]);
              }
            }}
          >
            <motion.div
              animate={{ y: [0, -5, 0] }}
              transition={{ repeat: Infinity, duration: 1.5 }}
            >
              <Upload className="w-8 h-8 text-gray-400 dark:text-gray-500 mb-2" />
            </motion.div>

            <p className="text-sm text-gray-500 dark:text-gray-400">
              Click to upload or drag & drop
            </p>

            <input
              id="file-upload"
              type="file"
              className="hidden"
              onChange={handleFileChange}
            />
          </motion.label>
        ) : (
          <motion.div
            key="file"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.25 }}
            className="flex items-center justify-between bg-gray-50 dark:bg-white/[0.05] p-3 rounded-lg"
          >
            <div className="flex items-center gap-3">
              <motion.div
                animate={{ rotate: [0, 5, -5, 0] }}
                transition={{ repeat: Infinity, duration: 2 }}
              >
                <File className="w-6 h-6 text-indigo-500" />
              </motion.div>

              <span className="text-sm text-gray-700 dark:text-gray-300">
                {file.name}
              </span>
            </div>

            <motion.button
              whileHover={{ scale: 1.1, rotate: 90 }}
              whileTap={{ scale: 0.9 }}
              onClick={removeFile}
              className="text-gray-400 hover:text-red-500"
            >
              <X className="w-5 h-5" />
            </motion.button>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
