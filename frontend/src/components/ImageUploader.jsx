export default function ImageUploader({ onChange }) {
  return (
    <input
      type="file"
      accept="image/png,image/jpeg,image/webp"
      onChange={(e) => onChange(e.target.files?.[0] || null)}
    />
  );
}
