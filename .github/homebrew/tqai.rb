class Tqai < Formula
  include Language::Python::Virtualenv

  desc "TurboQuant KV cache compression for local LLM inference"
  homepage "https://github.com/AlphaWaveSystems/tqai"
  url "https://files.pythonhosted.org/packages/source/t/tqai/tqai-0.1.0.tar.gz"
  sha256 "PLACEHOLDER_SHA256"
  license "MIT"

  depends_on "python@3.12"

  resource "numpy" do
    url "https://files.pythonhosted.org/packages/source/n/numpy/numpy-2.2.5.tar.gz"
    sha256 "PLACEHOLDER_SHA256"
  end

  def install
    virtualenv_install_with_resources
  end

  test do
    assert_match "tqai v", shell_output("#{bin}/tqai info")
  end
end

# NOTE: After publishing to PyPI, update the sha256 values:
#   1. Download the sdist: pip download tqai==0.1.0 --no-deps --no-binary :all:
#   2. Get sha256: shasum -a 256 tqai-0.1.0.tar.gz
#   3. Same for numpy
#
# To install this formula into the existing tap:
#   cp .github/homebrew/tqai.rb /path/to/homebrew-tap/Formula/tqai.rb
#   cd /path/to/homebrew-tap && git add Formula/tqai.rb && git commit -s -m "Add tqai formula"
