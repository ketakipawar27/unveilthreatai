import { useState, useEffect } from "react";
import { useModal } from "../../hooks/useModal";
import { Modal } from "../ui/modal";
import Button from "../ui/button/Button";
import Input from "../form/input/InputField";
import Label from "../form/Label";
import { motion, AnimatePresence } from "framer-motion";
import axios from "axios";

interface UserProfile {
  name: string;
  email: string;
  phone: string;
  avatar?: string;
}

export default function UserInfoCard() {
  const { isOpen, openModal, closeModal } = useModal();
  const [user, setUser] = useState<UserProfile>({
    name: "",
    email: "",
    phone: "",
    avatar: "/images/user/owner.jpg",
  });
  const [editingUser, setEditingUser] = useState<UserProfile>(user);
  const [avatarFile, setAvatarFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(true);

  // Fetch user info
  useEffect(() => {
    const fetchUserInfo = async () => {
      const token = localStorage.getItem("accessToken");
      if (!token) {
        setLoading(false);
        return;
      }

      try {
        const response = await axios.get("http://127.0.0.1:8000/api/user/profile/", {
          headers: { Authorization: `Bearer ${token}` },
        });

        const data = response.data;
        const fullName = `${data.first_name || ""} ${data.last_name || ""}`.trim();

        const fetchedUser = {
          name: fullName || data.username,
          email: data.email,
          phone: data.phone || "",
          avatar: data.avatar || "/images/user/owner.jpg",
        };

        setUser(fetchedUser);
        setEditingUser(fetchedUser);
      } catch (err) {
        console.error("Failed to fetch user info:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchUserInfo();
  }, []);

  // Image preview handler
  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setAvatarFile(file);
      const url = URL.createObjectURL(file);
      setEditingUser({ ...editingUser, avatar: url });
    }
  };

  const handleInputChange = (field: keyof UserProfile, value: string) => {
    setEditingUser({ ...editingUser, [field]: value });
  };

  // Save/update user info
  const handleSave = async () => {
    const token = localStorage.getItem("accessToken");
    if (!token) return;

    try {
      // Split full name into first_name and last_name
      const nameParts = editingUser.name.trim().split(" ");
      const first_name = nameParts[0] || "";
      const last_name = nameParts.slice(1).join(" ") || "";

      // Use FormData to send image
      const formData = new FormData();
      formData.append("first_name", first_name);
      formData.append("last_name", last_name);
      formData.append("phone", editingUser.phone);
      if (avatarFile) formData.append("avatar", avatarFile);

      const response = await axios.put(
        "http://127.0.0.1:8000/api/user/profile/",
        formData,
        {
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "multipart/form-data",
          },
        }
      );

      const updatedData = response.data;
      const fullName = `${updatedData.first_name || ""} ${updatedData.last_name || ""}`.trim();

      setUser({
        name: fullName || updatedData.username,
        email: updatedData.email,
        phone: updatedData.phone || "",
        avatar: updatedData.avatar || "/images/user/owner.jpg",
      });

      closeModal();
      setAvatarFile(null);
    } catch (err) {
      console.error("Failed to save user info:", err);
    }
  };

  if (loading) return <p>Loading user info...</p>;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="p-6 border border-gray-200 rounded-2xl shadow-sm dark:border-gray-800 lg:p-7 bg-white dark:bg-gray-900"
    >
      <div className="flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between">
        {/* Info Section */}
        <div>
          <h4 className="text-lg font-semibold text-gray-800 dark:text-white/90 mb-5">
            Personal Information
          </h4>

          <div className="grid grid-cols-1 gap-5 lg:grid-cols-2 2xl:gap-x-32">
            {/* Avatar */}
            <motion.div whileHover={{ scale: 1.05 }} className="flex flex-col items-center gap-3">
              <div className="w-20 h-20 overflow-hidden border-2 border-transparent rounded-full shadow-md transition-all duration-300 hover:border-brand-500">
                <img src={user.avatar} alt="user" className="w-full h-full object-cover" />
              </div>
              <p className="text-sm font-medium text-gray-800 dark:text-white/90">{user.name}</p>
            </motion.div>

            {/* Email */}
            <div>
              <p className="mb-1 text-xs text-gray-500 dark:text-gray-400">Email</p>
              <p className="text-sm font-medium text-gray-800 dark:text-white/90">{user.email}</p>
            </div>

            {/* Phone */}
            <div>
              <p className="mb-1 text-xs text-gray-500 dark:text-gray-400">Phone</p>
              <p className="text-sm font-medium text-gray-800 dark:text-white/90">{user.phone}</p>
            </div>
          </div>
        </div>

        {/* Edit Button */}
        <Button onClick={openModal} size="sm" className="self-start">
          Edit
        </Button>
      </div>

      {/* Modal */}
      <AnimatePresence>
        {isOpen && (
          <Modal isOpen={isOpen} onClose={closeModal} className="max-w-[700px] m-4">
            <motion.div
              initial={{ opacity: 0, scale: 0.9, y: 30 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.9, y: 30 }}
              transition={{ duration: 0.25 }}
              className="no-scrollbar relative w-full max-w-[700px] overflow-y-auto rounded-3xl bg-white p-6 dark:bg-gray-900 lg:p-11 shadow-xl"
            >
              <div className="px-2 pr-14">
                <h4 className="mb-2 text-2xl font-semibold text-gray-800 dark:text-white/90">
                  Edit Personal Information
                </h4>
                <p className="mb-6 text-sm text-gray-500 dark:text-gray-400">
                  Update your details to keep your profile up-to-date.
                </p>
              </div>

              <form className="flex flex-col gap-6">
                {/* Avatar Upload */}
                <div className="flex flex-col items-center gap-4">
                  <motion.div
                    whileHover={{ rotate: 3, scale: 1.05 }}
                    className="w-24 h-24 overflow-hidden border-2 border-gray-300 rounded-full shadow-md dark:border-gray-700"
                  >
                    <img src={editingUser.avatar} alt="user" className="w-full h-full object-cover" />
                  </motion.div>
                  <label className="px-3 py-1.5 text-sm bg-brand-500 text-white rounded-lg cursor-pointer hover:bg-brand-600 transition">
                    Change Photo
                    <input type="file" accept="image/*" className="hidden" onChange={handleImageChange} />
                  </label>
                </div>

                <div className="grid grid-cols-1 gap-x-6 gap-y-5 lg:grid-cols-2">
                  <div className="col-span-2">
                    <Label>Name</Label>
                    <Input
                      type="text"
                      value={editingUser.name}
                      onChange={(e) => handleInputChange("name", e.target.value)}
                    />
                  </div>

                  <div className="col-span-2 lg:col-span-1">
                    <Label>Email Address</Label>
                    <Input type="text" value={editingUser.email} disabled />
                  </div>

                  <div className="col-span-2 lg:col-span-1">
                    <Label>Phone</Label>
                    <Input
                      type="text"
                      value={editingUser.phone}
                      onChange={(e) => handleInputChange("phone", e.target.value)}
                    />
                  </div>
                </div>

                {/* Buttons */}
                <div className="flex items-center gap-3 mt-6 lg:justify-end">
                  <Button size="sm" variant="outline" onClick={closeModal} className="hover:scale-105 transition">
                    Close
                  </Button>
                  <Button size="sm" onClick={handleSave} className="hover:scale-105 transition">
                    Save Changes
                  </Button>
                </div>
              </form>
            </motion.div>
          </Modal>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
