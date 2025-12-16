interface FileInfoCardProps {
  fileName: string;
  fileType: string;
}

export default function FileInfoCard({ fileName, fileType }: FileInfoCardProps) {
  return (
    <div className="mt-6 overflow-hidden rounded-2xl border border-gray-200 bg-white p-6 
                    dark:border-gray-800 dark:bg-white/[0.03]">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-800 dark:text-white/90">
          Uploaded File Info
        </h3>
      </div>
      <div className="mt-4 space-y-2">
        <p className="text-sm text-gray-600 dark:text-gray-300">
          <span className="font-medium">File Name:</span> {fileName}
        </p>
        <p className="text-sm text-gray-600 dark:text-gray-300">
          <span className="font-medium">File Type:</span> {fileType || "Unknown"}
        </p>
      </div>
    </div>
  );
}
