use kornia_image::Image;

pub struct InfernumModelRequest {
    pub prompt: String,
    pub image: Image<u8, 3>,
}

pub struct InfernumModelResponse {
    pub response: String,
}

pub trait InfernumModel {
    type Error: std::error::Error + Send + Sync + 'static;

    fn run(&mut self, request: InfernumModelRequest) -> Result<InfernumModelResponse, Self::Error>;
}
