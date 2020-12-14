(defsystem "simple-nn"
  :version "0.1.0"
  :author ""
  :license ""
  :depends-on ()
  :components ((:module "src"
                :components
                ((:file "core"))))
  :description ""
  :in-order-to ((test-op (test-op "simple-nn/tests"))))

(defsystem "simple-nn/tests"
  :author ""
  :license ""
  :depends-on ("simple-nn"
               "rove")
  :components ((:module "tests"
                :components
                ((:file "core"))))
  :description "Test system for simple-nn"
  :perform (test-op (op c) (symbol-call :rove :run c)))
