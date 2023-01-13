
(cl:in-package :asdf)

(defsystem "custom_imagemsg-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :std_msgs-msg
)
  :components ((:file "_package")
    (:file "CustomImage" :depends-on ("_package_CustomImage"))
    (:file "_package_CustomImage" :depends-on ("_package"))
  ))